"""
Training entry point for nano-GPT JAX.

Run with::

    python train.py

On a machine with a GPU/TPU the full :func:`config.get_config` is used;
on CPU-only machines :func:`config.get_cpu_test_config` is selected automatically.
"""

import os

# Load environment variables from local .env if it exists, before JAX is imported.
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

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

from checkpoint import build_checkpoint_manager, restore_from_checkpoint, save_checkpoint
from config import get_config, get_cpu_test_config
from data import GPT2_VOCAB_SIZE, create_train_loader, create_val_loader
from model import (
    GPT,
    align_acc_step,
    apply_dtype_policy,
    dtype_report,
    is_cudnn_available,
    train_step,
    val_step,
)

print("jax.device_count():", jax.device_count())

nnx.use_eager_sharding(True)

# ── W&B setup ─────────────────────────────────────────────────────────────────


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


# ── Constants ──────────────────────────────────────────────────────────────────

_accelerator_backends = {"gpu", "tpu"}


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Training setup ──────────────────────────────────────────────────────────
    _has_accelerator = any(d.platform in _accelerator_backends for d in jax.devices())
    print(f"Training with accelerator: {_has_accelerator}")
    print("Is cudnn available? ", is_cudnn_available())

    cfg = get_config() if _has_accelerator else get_cpu_test_config()
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
        rngs = nnx.Rngs(0)
        model = GPT(cfg.model, rngs=rngs)

        if cfg.apply_dtype_policy:
            apply_dtype_policy(model, cfg.model)

        dtype_report(model)

        learning_rate = 6e-4
        decay_steps = cfg.max_steps - cfg.warmup_steps

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=decay_steps,
            end_value=learning_rate * 0.1,
        )

        def decay_mask(params):
            return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
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

        # ── Checkpointing ────────────────────────────────────────────────────────

        ckpt_mngr = build_checkpoint_manager(cfg)

        # Resume from checkpoint if configured.
        start_step = restore_from_checkpoint(cfg, ckpt_mngr, model, optimizer, train_iter)

        # ── Training loop ────────────────────────────────────────────────────────

        max_steps = cfg.max_steps
        last_val_loss = float("inf")  # track for checkpointing metrics

        run = wandb.init(
            project="nano-gpt-jax",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "n_layer": cfg.model.n_layer,
                "n_head": cfg.model.n_head,
                "n_embd": cfg.model.n_embd,
                "block_size": cfg.model.block_size,
                "vocab_size": cfg.model.vocab_size,
                "batch_size": cfg.batch_size,
                "sequence_length": cfg.sequence_length,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "dtype_policy": cfg.apply_dtype_policy,
            },
        )

        t0 = time.time()
        for micro_step, batch in enumerate(
            train_iter, start=start_step * cfg.grad_acc_steps
        ):
            if micro_step >= max_steps * cfg.grad_acc_steps:
                break

            # Validation evaluation / Checkpointing
            if micro_step % (cfg.val_check_steps * cfg.grad_acc_steps) == 0:
                val_loss = last_val_loss
                if val_loader is not None:
                    t_val_start = time.time()
                    val_loss_accum = 0.0
                    val_steps = 0

                    # Each iter() call restarts from the beginning of the val split
                    # (num_epochs=1 in the val loader).
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
                        f"step {align_acc_step(micro_step, cfg.grad_acc_steps):4d} | validation loss {val_loss:.4f} | val_time {val_dt * 1000:.2f} ms | val_tokens/sec {val_tokens_per_sec:.2f}"
                    )
                    run.log(
                        {
                            "val_loss": val_loss,
                            "val_time_ms": val_dt * 1000,
                            "val_tokens_per_sec": val_tokens_per_sec,
                        },
                        step=align_acc_step(micro_step, cfg.grad_acc_steps),
                    )

                # ── Checkpoint (async) ───────────────────────────────────────
                eff_step = align_acc_step(micro_step, cfg.grad_acc_steps)
                if eff_step > 0 and eff_step % cfg.ckpt_every_steps == 0:
                    wandb_run = run if not isinstance(wandb, MockWandb) else None
                    save_checkpoint(
                        ckpt_mngr, model, optimizer, eff_step,
                        val_loss, cfg, train_iter, wandb_run,
                    )

                # Reset timer so validation time doesn't pollute training throughput
                t0 = time.time()

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
                total_tokens = (
                    micro_step
                    * cfg.grad_acc_steps
                    * cfg.sequence_length
                    * cfg.batch_size
                )
                run.log(
                    {
                        "loss": loss.item(),
                        "step_time_ms": dt * 1000,
                        "tokens_per_sec": tokens_per_sec,
                        "learning_rate": schedule(optimizer.step[...]).item(),
                        "total_tokens": total_tokens,
                    },
                    step=align_acc_step(micro_step, cfg.grad_acc_steps),
                )
                print(
                    f"step {align_acc_step(micro_step, cfg.grad_acc_steps):4d} | loss {loss:.4f} | lr: {schedule(optimizer.step[...]):.4f} | time {dt * 1000:.2f} ms | tokens/sec {tokens_per_sec:.2f} | tokens {total_tokens}"
                )
                t0 = time.time()

        # ── Cleanup ──────────────────────────────────────────────────────────────
        ckpt_mngr.wait_until_finished()
        ckpt_mngr.close()
        run.finish()
