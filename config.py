"""
Training configuration for nano-GPT JAX.

Provides three configs:
- :func:`get_config`: full GPU training configuration.
- :func:`get_cpu_test_config`: lightweight CPU-only configuration for quick iteration.
- :func:`get_mem_eff_config`: like ``get_cpu_test_config`` but uses :class:`~attention.MemoryEfficientAttention`.
"""

import os

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.sequence_length = 1024
    # Mixed precision: use bfloat16 for faster computation, weights remain float32,
    # activations are bfloat16. Don't apply to LayerNorm as it sums many values —
    # this can lead to overflow or underflow.
    cfg.apply_dtype_policy = True
    cfg.num_devices = jax.device_count()
    cfg.device_batch_size = 32
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.gpt_batch_size = 524288
    cfg.grad_acc_steps = cfg.gpt_batch_size // (cfg.batch_size * cfg.sequence_length)
    cfg.dataset = "edu_fineweb"  # "input_txt" or "edu_fineweb"
    cfg.val_check_steps = 100  # evaluate validation loss every 100 steps
    cfg.val_max_steps = 50  # max number of batches to use for validation
    cfg.max_steps = 10_000  # total training steps
    cfg.warmup_steps = 715

    # Checkpointing config
    cfg.ckpt_dir_name = "checkpoints"
    cfg.ckpt_dir = os.path.join(os.path.dirname(__file__), cfg.ckpt_dir_name)
    cfg.ckpt_every_steps = (
        cfg.val_check_steps * 1
    )  # save every N effective training steps; best aligned with validation checks
    cfg.ckpt_max_to_keep = 3  # keep N most recent + best val_loss
    cfg.resume_ckpt = None  # step number, "latest", or None to start fresh

    # GPT model config
    cfg.model = ConfigDict()
    cfg.model.block_size = cfg.sequence_length
    cfg.model.vocab_size = -1
    cfg.model.n_layer = 6
    cfg.model.n_head = 6
    cfg.model.n_embd = 384
    cfg.model.param_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.compute_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.accum_dtype = jnp.float32

    # Attention implementation: "flash" | "flax" | "classical"
    # See attention.py / AttentionType for details.
    cfg.model.attention_type = "flash"  # cuDNN on GPU, XLA fallback on CPU
    cfg.use_attention_bias = True

    return cfg


def get_cpu_test_config() -> ConfigDict:
    """Lightweight config for quick testing on CPU."""
    cfg = get_config()
    cfg.apply_dtype_policy = False
    cfg.device_batch_size = 1
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.gpt_batch_size = cfg.batch_size
    cfg.grad_acc_steps = 1
    cfg.val_check_steps = 10  # evaluate validation loss every 10 steps
    cfg.val_max_steps = 4  # max number of batches to use for validation
    cfg.max_steps = 200  # total training steps
    cfg.warmup_steps = 10
    cfg.resume_ckpt = None  # step number, "latest", or None to start fresh

    cfg.model.n_layer = 3
    cfg.model.n_head = 2
    cfg.model.n_embd = 16

    cfg.dataset = "input_txt"

    # Use the Flax reference implementation on CPU so there is a known-good
    # baseline to compare against when developing custom attention.
    cfg.model.attention_type = "flax"

    return cfg


def get_mem_eff_config() -> ConfigDict:
    """CPU config that uses :class:`~attention.MemoryEfficientAttention`.

    Inherits all settings from :func:`get_cpu_test_config` and overrides
    ``attention_type`` to ``"mem_eff"``.  Chunk sizes are set to 64 tokens
    by default — tweak them to trade compilation time for memory savings.
    """
    cfg = get_cpu_test_config()
    cfg.model.attention_type = "mem_eff"

    # How many query / key tokens to process per chunk.
    # Smaller values → less peak memory, more XLA loop iterations.
    cfg.model.query_chunk_size = 64
    cfg.model.key_chunk_size = 64

    return cfg
