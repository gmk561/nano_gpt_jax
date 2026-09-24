"""
Training configuration for nano-GPT JAX.

Provides a single :func:`get_config` factory with two presets:

- ``"gpu"`` — full training on accelerators (flash attention, bfloat16, Muon
  optimizer, EduFineweb dataset).
- ``"cpu"`` — lightweight testing on CPU (flax attention, float32, AdamW
  optimizer, input.txt dataset).

Pass ``preset="auto"`` (the default) to auto-detect based on available JAX
devices.

Examples::

    cfg = get_config()            # auto-detect
    cfg = get_config("gpu")       # force GPU settings
    cfg = get_config("cpu")       # force CPU settings
"""

import os

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Valid optimizer types (plain strings, no enum).
OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_MUON = "muon"

# Valid LR schedule types.
LR_SCHEDULE_COSINE = "cosine"
LR_SCHEDULE_TRAPEZOIDAL = "trapezoidal"

_ACCELERATOR_BACKENDS = {"gpu", "tpu"}


def _has_accelerator() -> bool:
    """Return True if any JAX device is a GPU or TPU."""
    return any(d.platform in _ACCELERATOR_BACKENDS for d in jax.devices())


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def get_config(preset: str = "auto") -> ConfigDict:
    """Build a training :class:`ConfigDict`.

    Parameters
    ----------
    preset:
        ``"gpu"`` for full GPU training, ``"cpu"`` for lightweight CPU testing,
        or ``"auto"`` (default) to choose based on detected JAX devices.

    Returns
    -------
    ConfigDict
        Complete training configuration.
    """
    if preset == "auto":
        preset = "gpu" if _has_accelerator() else "cpu"

    if preset not in ("gpu", "cpu"):
        raise ValueError(f"Unknown preset {preset!r}. Expected 'gpu', 'cpu', or 'auto'.")

    is_gpu = preset == "gpu"

    cfg = ConfigDict()

    # ── Optimizer ───────────────────────────────────────────────────────────
    cfg.optimizer = ConfigDict()
    cfg.optimizer.type = OPTIMIZER_MUON if is_gpu else OPTIMIZER_ADAMW

    # Muon hyperparameters
    cfg.optimizer.momentum = 0.95
    cfg.optimizer.muon_weight_decay = 0.0

    # AdamW hyperparameters (standalone, or for non-matrix params under Muon)
    cfg.optimizer.adam_learning_rate = 3e-4
    cfg.optimizer.b1 = 0.90
    cfg.optimizer.b2 = 0.95
    cfg.optimizer.eps = 1e-8
    cfg.optimizer.weight_decay = 0.01
    cfg.optimizer.clip_by_global_norm = None

    # ── Mixed precision ────────────────────────────────────────────────────
    cfg.is_gpu = is_gpu
    # bfloat16 activations/params on GPU; full float32 on CPU.
    cfg.apply_dtype_policy = is_gpu

    # ── Data ───────────────────────────────────────────────────────────────
    cfg.sequence_length = 1024
    cfg.num_devices = jax.device_count()
    cfg.device_batch_size = 64 if is_gpu else 1
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.dataset = "edu_fineweb" if is_gpu else "input_txt"

    if is_gpu:
        cfg.tokens_per_batch = 524288
        cfg.grad_acc_steps = cfg.tokens_per_batch // (cfg.batch_size * cfg.sequence_length)
    else:
        cfg.tokens_per_batch = cfg.batch_size * cfg.sequence_length
        cfg.grad_acc_steps = 1

    # ── Training schedule ──────────────────────────────────────────────────
    cfg.max_steps = 12000 if is_gpu else 200
    cfg.warmup_steps = int(cfg.max_steps * 0.2) if is_gpu else 10
    cfg.val_check_steps = 500 if is_gpu else 10
    cfg.val_max_steps = 50 if is_gpu else 4
    cfg.learning_rate = 0.02
    cfg.lr_end_ratio = 0.1  # end_value = learning_rate * lr_end_ratio
    cfg.lr_schedule = LR_SCHEDULE_TRAPEZOIDAL
    cfg.seed = 0

    # ── Checkpointing ──────────────────────────────────────────────────────
    cfg.ckpt_dir_name = "checkpoints"
    cfg.ckpt_dir = os.path.join(os.path.dirname(__file__), cfg.ckpt_dir_name)
    cfg.ckpt_every_steps = cfg.val_check_steps * 1000
    cfg.ckpt_max_to_keep = 3
    cfg.resume_ckpt = None  # step number, "latest", or None to start fresh

    # ── GPT model ──────────────────────────────────────────────────────────
    cfg.model = ConfigDict()
    cfg.model.max_seq_len = cfg.sequence_length
    cfg.model.vocab_size = -1  # set at runtime from tokenizer

    if is_gpu:
        cfg.model.n_layer = 12
        cfg.model.n_head = 12
        cfg.model.n_embd = 768
    else:
        cfg.model.n_layer = 3
        cfg.model.n_head = 2
        cfg.model.n_embd = 16

    cfg.model.param_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.compute_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.accum_dtype = jnp.float32

    # Attention: "flash" on GPU (cuDNN/XLA), "flax" on CPU (reference).
    cfg.model.attention_type = "flash" if is_gpu else "flax"

    return cfg
