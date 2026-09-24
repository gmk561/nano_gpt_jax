"""
Checkpointing helpers for nano-GPT JAX.

Manages Orbax checkpoint creation/restoration and Grain iterator state
persistence alongside each checkpoint.

Functions
---------
build_checkpoint_manager(cfg)
    Create and return an :class:`~orbax.checkpoint.CheckpointManager`.
get_checkpoint_state(model, optimizer, step)
    Extract a checkpoint-friendly PyTree from NNX modules.
restore_checkpoint_state(model, optimizer, state_dict)
    Load a PyTree back into NNX modules; returns the saved step.
save_checkpoint(ckpt_mngr, model, optimizer, step, val_loss, cfg, train_iter, wandb_run)
    Save Orbax checkpoint + Grain iterator state + optional W&B artifact.
restore_from_checkpoint(cfg, ckpt_mngr, model, optimizer, train_iter)
    Restore model/optimizer state and Grain iterator from the configured
    checkpoint; returns the step to resume from.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import orbax.checkpoint as ocp
from flax import nnx
from ml_collections import ConfigDict

from data import get_iter_state, restore_iter_state

if TYPE_CHECKING:
    from model import GPT


# ── Manager factory ────────────────────────────────────────────────────────────


def build_checkpoint_manager(cfg: "ConfigDict") -> ocp.CheckpointManager:
    """Create an Orbax :class:`~orbax.checkpoint.CheckpointManager` from config."""
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.ckpt_max_to_keep,
        best_fn=lambda metrics: metrics["val_loss"],
        best_mode="min",
        enable_async_checkpointing=True,
    )
    return ocp.CheckpointManager(cfg.ckpt_dir, options=options)


# ── State helpers ──────────────────────────────────────────────────────────────


def get_checkpoint_state(
    model: "GPT",
    optimizer: nnx.Optimizer,
    step: int,
) -> dict:
    """Extract a checkpoint-friendly PyTree from NNX modules.

    Data-loader position is **not** stored here; it is written separately as a
    Grain state file (``grain_train_state.bin``) alongside each Orbax
    checkpoint directory.  Use :func:`data.get_iter_state` /
    :func:`data.restore_iter_state` for that.
    """
    return {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
    }


def restore_checkpoint_state(
    model: "GPT",
    optimizer: nnx.Optimizer,
    state_dict: dict,
) -> int:
    """Restore NNX module state from a loaded checkpoint PyTree.

    Returns
    -------
    int
        The effective training step stored in the checkpoint.
    """
    nnx.update(model, state_dict["model"])
    nnx.update(optimizer, state_dict["optimizer"])
    return int(state_dict["step"])


# ── Save / restore ─────────────────────────────────────────────────────────────


def save_checkpoint(
    ckpt_mngr: ocp.CheckpointManager,
    model: "GPT",
    optimizer: nnx.Optimizer,
    step: int,
    val_loss: float,
    cfg: "ConfigDict",
    train_iter,
    wandb_run=None,
) -> None:
    """Save model + optimizer state, Grain iterator position, and optional W&B artifact.

    Parameters
    ----------
    ckpt_mngr:
        Active :class:`~orbax.checkpoint.CheckpointManager`.
    model:
        The NNX GPT model.
    optimizer:
        The NNX optimizer.
    step:
        Effective training step to label this checkpoint.
    val_loss:
        Validation loss at this step (stored as Orbax metric for best-checkpoint tracking).
    cfg:
        Training configuration (used for ``ckpt_dir``).
    train_iter:
        Active Grain :class:`~grain.python.DataLoaderIterator` to snapshot.
    wandb_run:
        Optional W&B run object; if provided, the checkpoint is uploaded as an artifact.
    """
    ckpt_state = get_checkpoint_state(model, optimizer, step)
    saved = ckpt_mngr.save(
        step,
        args=ocp.args.StandardSave(ckpt_state),
        metrics={"val_loss": val_loss},
    )
    if not saved:
        print(f"  → checkpoint at step {step} skipped by manager")
        return

    print(f"  → checkpoint saved at step {step}")

    # Wait for the async Orbax write so the step directory exists before we
    # write the Grain state file next to it.
    ckpt_mngr.wait_until_finished()

    # Persist Grain iterator state alongside the checkpoint.
    step_dir = os.path.join(cfg.ckpt_dir, str(step))
    os.makedirs(step_dir, exist_ok=True)
    grain_state_path = os.path.join(step_dir, "grain_train_state.bin")
    with open(grain_state_path, "wb") as fh:
        fh.write(get_iter_state(train_iter))

    # Optionally upload to W&B as an artifact — done in a background thread
    # so it does not block training between checkpoints.
    if wandb_run is not None:
        ckpt_path = os.path.join(cfg.ckpt_dir, str(step))

        def _upload(run=wandb_run, path=ckpt_path, s=step):
            try:
                import wandb

                artifact = wandb.Artifact(f"checkpoint-step-{s}", type="model")
                if os.path.isdir(path):
                    artifact.add_dir(path)
                    run.log_artifact(artifact)
            except Exception as e:
                print(f"  ⚠ W&B artifact upload failed: {e}")

        threading.Thread(target=_upload, daemon=True).start()


def restore_from_checkpoint(
    cfg: "ConfigDict",
    ckpt_mngr: ocp.CheckpointManager,
    model: "GPT",
    optimizer: nnx.Optimizer,
    train_iter,
) -> int:
    """Restore model/optimizer and Grain iterator from the configured checkpoint.

    Parameters
    ----------
    cfg:
        Training configuration.  ``cfg.resume_ckpt`` controls what to restore:
        ``"latest"`` picks the most recent step, an integer picks a specific
        step, and ``None`` means no restoration.
    ckpt_mngr:
        Active :class:`~orbax.checkpoint.CheckpointManager`.
    model:
        NNX GPT model (will be updated in-place).
    optimizer:
        NNX optimizer (will be updated in-place).
    train_iter:
        Freshly created Grain :class:`~grain.python.DataLoaderIterator`
        (must not have produced any elements yet).

    Returns
    -------
    int
        The step to resume training from (0 if no checkpoint was restored).
    """
    resume = cfg.resume_ckpt
    if resume is None:
        return 0

    restore_step = ckpt_mngr.latest_step() if resume == "latest" else int(resume)
    if restore_step is None:
        print("No checkpoint found to resume from, starting fresh")
        return 0

    abstract_state = get_checkpoint_state(model, optimizer, 0)
    restored = ckpt_mngr.restore(
        restore_step,
        args=ocp.args.StandardRestore(abstract_state),
    )
    start_step = restore_checkpoint_state(model, optimizer, restored)
    print(f"Resumed from checkpoint at step {start_step}")

    # Restore Grain iterator state if the companion file exists.
    grain_state_path = os.path.join(
        cfg.ckpt_dir, str(restore_step), "grain_train_state.bin"
    )
    if os.path.exists(grain_state_path):
        with open(grain_state_path, "rb") as fh:
            restore_iter_state(train_iter, fh.read())
        print("  → Grain data iterator restored from checkpoint")
    else:
        print("  → no Grain iterator state found; data starts from epoch beginning")

    return start_step
