"""
Training entry point for nano-GPT JAX.

Run with::

    python train.py [--preset gpu|cpu] [--set KEY=VALUE ...]

Examples::

    python train.py                                 # auto-detect GPU/CPU
    python train.py --preset cpu --set max_steps=50
    python train.py --set optimizer.type=muon
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
import wandb
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from checkpoint import (
    build_checkpoint_manager,
    restore_from_checkpoint,
    save_checkpoint,
)
from config import OPTIMIZER_MUON, get_config
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
from optimizer import build_lr_schedule, build_optimizer, print_optimizer_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nano-GPT JAX training")
    parser.add_argument(
        "--preset",
        choices=["gpu", "cpu", "auto"],
        default="auto",
        help="Config preset. Defaults to 'auto' (GPU if accelerator detected, else CPU).",
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


if __name__ == "__main__":
    args = _parse_args()

    # ── Training setup ──────────────────────────────────────────────────────
    print("jax.device_count():", jax.device_count())
    nnx.use_eager_sharding(True)

    cfg = get_config(preset=args.preset)
    _apply_overrides(cfg, args.set)

    is_gpu = cfg.is_gpu
    print(f"Using preset: {'gpu' if is_gpu else 'cpu'}")
    print("Is cudnn available?", is_cudnn_available())

    # 2D mesh: ('data', 'model'). For now model=1; later for tensor parallelism.
    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))

    # Build Grain data loaders.
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

        optimizer, schedule = build_optimizer(model, cfg)
        print_optimizer_params(model, cfg)

        is_muon = cfg.optimizer.type == OPTIMIZER_MUON
        if is_muon:
            muon_schedule = schedule
            adam_schedule = build_lr_schedule(cfg, learning_rate=cfg.optimizer.adam_learning_rate)
        else:
            adam_schedule = schedule

        ckpt_mngr = build_checkpoint_manager(cfg)

        # Resume from checkpoint if configured.
        start_step = restore_from_checkpoint(
            cfg, ckpt_mngr, model, optimizer, train_iter
        )
        print("Start step:", start_step)

        last_val_loss = float("inf")

        run = wandb.init(
            project="nano-gpt-jax",
            mode=os.environ.get("WANDB_MODE", "online"),
            config=cfg.to_dict(),
        )

        t0 = time.time()
        for micro_step, batch in enumerate(
            train_iter, start=start_step * cfg.grad_acc_steps
        ):
            global_step = align_acc_step(micro_step, cfg.grad_acc_steps)

            is_last_step = micro_step >= cfg.max_steps * cfg.grad_acc_steps
            if (
                micro_step % (cfg.val_check_steps * cfg.grad_acc_steps) == 0
                or is_last_step
            ):
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
                    wandb_run = run if not getattr(run, "disabled", False) else None
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
                total_tokens = micro_step * cfg.sequence_length * cfg.batch_size
                log_dict = {
                    "loss": loss.item(),
                    "step_time_ms": dt * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "total_tokens_b": total_tokens / 1e9,
                }
                if is_muon:
                    log_dict["learning_rate/muon"] = muon_schedule(global_step).item()
                    log_dict["learning_rate/adam"] = adam_schedule(global_step).item()
                    lr_str = f"muon_lr: {muon_schedule(global_step):.4f} | adam_lr: {adam_schedule(global_step):.6f}"
                else:
                    log_dict["learning_rate"] = adam_schedule(global_step).item()
                    lr_str = f"lr: {adam_schedule(global_step):.6f}"

                run.log(log_dict, step=global_step)

                print(
                    f"step {global_step:4d} | loss {loss:.4f} | {lr_str} | time {dt * 1000:.2f} ms | tokens/sec {tokens_per_sec:.2f} | tokens {total_tokens / 1e9:.3f}B"
                )
                t0 = time.time()

        ckpt_mngr.wait_until_finished()
        ckpt_mngr.close()
        run.finish()
