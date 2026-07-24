"""
Training entry point for nano-GPT JAX.

Run with::

    python train.py [--config gpu|cpu|mem_eff] [--set KEY=VALUE ...]

Examples::

    python train.py --config mem_eff --set model.query_chunk_size=32
    python train.py --config gpu --set model.attention_type=classical
    python train.py --set resume_ckpt=latest
"""

import argparse
import os

# Load environment variables from local .env if it exists, before JAX is imported.
# Handles `export KEY=VALUE`, quoted values, and comment lines.
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip optional `export ` prefix
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, _, val = line.partition("=")
            # Strip surrounding single or double quotes from the value
            val = val.strip().strip("'\"")
            os.environ.setdefault(key.strip(), val)

# Emulate 8 CPU devices by default if no XLA_FLAGS are set (uncomment to use):
# if "XLA_FLAGS" not in os.environ:
#     os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from attention import is_cudnn_available
from checkpoint import (
    build_checkpoint_manager,
    restore_from_checkpoint,
    save_checkpoint,
)
from config import LRSchedule, get_config, get_cpu_test_config, get_mem_eff_config
from data import GPT2_VOCAB_SIZE, create_train_loader, create_val_loader
from model import (
    GPT,
    align_acc_step,
    apply_dtype_policy,
    dtype_report,
    train_step,
    val_step,
)


_CONFIGS = {
    "gpu": get_config,
    "cpu": get_cpu_test_config,
    "mem_eff": get_mem_eff_config,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-GPT JAX training")
    parser.add_argument(
        "--config",
        choices=list(_CONFIGS),
        default=None,
        help="Config preset. Defaults to 'gpu' if an accelerator is detected, else 'cpu'.",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Override a config field, e.g. --set model.n_layer=12 --set max_steps=5000.",
    )
    return parser.parse_args()


def _apply_overrides(cfg, overrides: list[str]) -> None:
    """Apply KEY=VALUE overrides to the ConfigDict."""
    for kv in overrides:
        key, _, raw = kv.partition("=")
        # Try to cast to int, float, or bool; fall back to raw string.
        value: object = raw
        for cast in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                value = cast(raw)
                break
            except (ValueError, KeyError):
                pass
        # Traverse nested keys: "model.n_layer" → cfg.model.n_layer
        parts = key.strip().split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


class MockWandb:
    @staticmethod
    def init(*args, **kwargs):
        class Run:
            def log(self, *args, **kwargs):
                pass

            def finish(self, *args, **kwargs):
                pass

            def log_artifact(self, *args, **kwargs):
                pass

        return Run()

    @staticmethod
    def log(*args, **kwargs):
        pass

    @staticmethod
    def finish(*args, **kwargs):
        pass


if os.environ.get("WANDB_MODE") == "disabled":
    wandb = MockWandb()
else:
    import wandb  # type: ignore[no-redef]


_accelerator_backends = {"gpu", "tpu"}


def trapezoidal_schedule(config):
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    plateau_schedule = optax.constant_schedule(value=config.learning_rate)
    decay_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        transition_steps=config.warmup_steps,
        end_value=0.0,
    )

    return optax.join_schedules(
        schedules=[warmup_schedule, plateau_schedule, decay_schedule],
        boundaries=[config.warmup_steps, config.max_steps - config.warmup_steps],
    )


if __name__ == "__main__":
    args = _parse_args()

    # ── Training setup ──────────────────────────────────────────────────────────
    print("jax.device_count():", jax.device_count())
    nnx.use_eager_sharding(True)

    _has_accelerator = any(d.platform in _accelerator_backends for d in jax.devices())
    print(f"Training with accelerator: {_has_accelerator}")
    print("Is cudnn available? ", is_cudnn_available())

    preset = args.config or ("gpu" if _has_accelerator else "cpu")
    cfg = _CONFIGS[preset]()
    _apply_overrides(cfg, args.set)
    print(f"Using config preset: {preset}")
    # 2D mesh: ('data', 'model'). For now model=1; later change to (dp, mp) for tensor parallelism.
    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))

    # Build Grain data loaders (support multi-host data parallelism via
    # ShardByJaxProcess; on a single process this is equivalent to NoSharding).
    train_loader = create_train_loader(cfg)
    val_loader = create_val_loader(cfg)  # None for input_txt dataset

    # Both datasets use the GPT-2 tokenizer.
    cfg.model.vocab_size = GPT2_VOCAB_SIZE

    # Create a persistent training iterator so we can checkpoint its position.
    train_iter = iter(train_loader)

    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(cfg.seed)
        model = GPT(cfg.model, rngs=rngs)

        if cfg.apply_dtype_policy:
            apply_dtype_policy(model, cfg.model)

        dtype_report(model)

        if cfg.lr_schedule == LRSchedule.COSINE:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=cfg.learning_rate,
                warmup_steps=cfg.warmup_steps,
                decay_steps=cfg.max_steps - cfg.warmup_steps,
                end_value=cfg.learning_rate * cfg.lr_end_ratio,
            )
        elif cfg.lr_schedule == LRSchedule.TRAPEZOIDAL:
            schedule = trapezoidal_schedule(cfg)
        else:
            raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")

        def decay_mask(params):
            return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)

        tx = optax.chain(
            # Remove gradient clipping for faster gradient propagation.
            # optax.clip_by_global_norm(1.0),
            optax.adamw(
                schedule,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=0.1,
                mask=decay_mask,
            ),
        )

        if cfg.grad_acc_steps > 1:
            tx = optax.MultiSteps(
                tx, every_k_schedule=cfg.grad_acc_steps, use_grad_mean=True
            )

        optimizer = nnx.Optimizer(
            model,
            tx,
            wrt=nnx.Param,
        )

        ckpt_mngr = build_checkpoint_manager(cfg)

        # Resume from checkpoint if configured.
        start_step = restore_from_checkpoint(
            cfg, ckpt_mngr, model, optimizer, train_iter
        )

        last_val_loss = float("inf")  # track for checkpointing metrics

        run = wandb.init(
            project="nano-gpt-jax",
            mode=os.environ.get("WANDB_MODE", "online"),
            config=cfg.to_dict(),
        )

        for micro_step, batch in enumerate(
            train_iter, start=start_step * cfg.grad_acc_steps
        ):
            global_step = align_acc_step(micro_step, cfg.grad_acc_steps)

            is_last_step = micro_step >= cfg.max_steps * cfg.grad_acc_steps
            if micro_step % (cfg.val_check_steps * cfg.grad_acc_steps) == 0 or is_last_step:
                val_loss = last_val_loss
                if val_loader is not None:
                    t_val_start = time.time()
                    val_loss_accum = 0.0
                    val_steps = 0

                    for val_batch in iter(val_loader):
                        if val_steps >= cfg.val_max_steps:
                            break
                        # Shard validation inputs along the data dimension
                        x_val_sharded = jax.device_put(
                            val_batch["x"], NamedSharding(mesh, P("data", None))
                        )
                        y_val_sharded = jax.device_put(
                            val_batch["y"], NamedSharding(mesh, P("data", None))
                        )
                        v_loss = val_step(model, x_val_sharded, y_val_sharded)
                        val_loss_accum += v_loss.item()
                        val_steps += 1

                    t_val_end = time.time()
                    val_dt = t_val_end - t_val_start

                    total_val_tokens = val_steps * cfg.sequence_length * cfg.batch_size
                    val_tokens_per_sec = (
                        total_val_tokens / val_dt if val_dt > 0 else 0.0
                    )

                    val_loss = (
                        val_loss_accum / val_steps if val_steps > 0 else float("inf")
                    )
                    last_val_loss = val_loss
                    print(
                        f"step {global_step:4d} | validation loss {val_loss:.4f} | val_time {val_dt * 1000:.2f} ms | val_tokens/sec {val_tokens_per_sec:.2f}"
                    )
                    run.log(
                        {
                            "val_loss": val_loss,
                            "val_time_ms": val_dt * 1000,
                            "val_tokens_per_sec": val_tokens_per_sec,
                        },
                        step=global_step,
                    )

                if global_step > 0 and global_step % cfg.ckpt_every_steps == 0:
                    wandb_run = run if not isinstance(wandb, MockWandb) else None
                    save_checkpoint(
                        ckpt_mngr,
                        model,
                        optimizer,
                        global_step,
                        val_loss,
                        cfg,
                        train_iter,
                        wandb_run,
                    )

                # Reset timer so validation time doesn't pollute training throughput
                t0 = time.time()

            if is_last_step:
                break

            # Shard training batch along the data mesh dimension
            x_sharded = jax.device_put(batch["x"], NamedSharding(mesh, P("data", None)))
            y_sharded = jax.device_put(batch["y"], NamedSharding(mesh, P("data", None)))
            loss = train_step(model, optimizer, x_sharded, y_sharded)
            loss.block_until_ready()  # ensure device work is done before timing

            if micro_step % cfg.grad_acc_steps == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    cfg.sequence_length * cfg.batch_size * cfg.grad_acc_steps / dt
                )
                # micro_step already counts every micro-batch, and each
                # processes batch_size * sequence_length tokens.
                total_tokens = micro_step * cfg.sequence_length * cfg.batch_size
                run.log(
                    {
                        "loss": loss.item(),
                        "step_time_ms": dt * 1000,
                        "tokens_per_sec": tokens_per_sec,
                        "learning_rate": schedule(global_step).item(),
                        "total_tokens_b": total_tokens / 1e9,
                    },
                    step=global_step,
                )
                print(
                    f"step {global_step:4d} | loss {loss:.4f} | lr: {schedule(global_step):.4f} | time {dt * 1000:.2f} ms | tokens/sec {tokens_per_sec:.2f} | tokens {total_tokens / 1e9:.3f}B"
                )
                t0 = time.time()

        ckpt_mngr.wait_until_finished()
        ckpt_mngr.close()
        run.finish()
