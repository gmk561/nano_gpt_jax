"""
Attention mechanisms for nano-GPT JAX.

Provides multiple multi-head attention implementations selectable via
:class:`AttentionType` and the :func:`build_attention_module` factory.

AttentionType
-------------
FLAX
    The out-of-the-box ``nnx.MultiHeadAttention`` with its default built-in
    attention kernel.  Use this as a numerically stable reference baseline
    when developing and debugging custom attention variants.
FLASH
    ``nnx.MultiHeadAttention`` with ``jax.nn.dot_product_attention`` as the
    inner kernel.  Dispatches to cuDNN flash attention on supported GPUs and
    falls back to the XLA implementation otherwise.  This is the current
    production default.
CLASSICAL
    Explicit scaled dot-product attention written from scratch following
    Vaswani et al. (2017) "Attention is All You Need".  Useful as a learning
    exercise and for building new attention variants on top of a clean,
    readable reference.

Configure via ``cfg.model.attention_type`` in ``config.py``.
"""

from __future__ import annotations

import functools
import math
from enum import Enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from ml_collections import ConfigDict


# ── Enum ───────────────────────────────────────────────────────────────────────


class AttentionType(str, Enum):
    """Selects the multi-head attention implementation used in each transformer block.

    Use a plain string value in ``cfg.model.attention_type``; the enum is used
    internally for exhaustive matching.

    Values
    ------
    ``"flax"``
        Reference: ``nnx.MultiHeadAttention`` with Flax's default kernel.
    ``"flash"``
        Production: ``nnx.MultiHeadAttention`` + JAX flash/XLA kernel.
    ``"classical"``
        Exercise: explicit SDPA from "Attention is All You Need".
    """

    FLAX = "flax"
    FLASH = "flash"
    CLASSICAL = "classical"


# ── cuDNN detection ────────────────────────────────────────────────────────────


def is_cudnn_available() -> bool:
    """Return ``True`` if a cuDNN runtime is accessible via JAX."""
    try:
        from jax._src.lib import cuda_versions

        return (
            cuda_versions is not None and cuda_versions.cudnn_get_version() is not None
        )
    except (ImportError, AttributeError, RuntimeError):
        return False


# ── Inner attention kernels ────────────────────────────────────────────────────


def _flash_attention_kernel(
    query, key, value, bias=None, mask=None, is_causal: bool = True, **kwargs
):
    """JAX ``dot_product_attention`` dispatching to cuDNN when available.

    This function is passed as ``attention_fn`` to ``nnx.MultiHeadAttention``
    and handles only the core QK^T V computation — the QKV/output projections
    are still managed by ``nnx.MultiHeadAttention``.
    """
    impl = "cudnn" if is_cudnn_available() else "xla"
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask if not is_causal else None,
        is_causal=is_causal,
        implementation=impl,
    )


# ── Classical multi-head attention ─────────────────────────────────────────────


class ClassicalMultiHeadAttention(nnx.Module):
    """Multi-head attention from "Attention is All You Need" (Vaswani et al., 2017).

    Implements scaled dot-product attention explicitly without relying on any
    JAX-level attention kernel optimisation.  Every operation is written out
    so the implementation can serve as a readable, step-by-step reference:

    .. code-block:: text

        Q = x W_Q,  K = x W_K,  V = x W_V     # linear projections
        scores = Q K^T / sqrt(d_k)              # scale
        scores = masked_fill(scores, mask==0, -inf)  # causal mask
        weights = softmax(scores, dim=-1)       # attention weights
        context = weights V                     # aggregate values
        out = context W_O                       # output projection

    Parameters
    ----------
    config:
        Model config; must provide ``n_head``, ``n_embd``, ``compute_dtype``.
    rngs:
        NNX PRNG key bundle.
    """

    def __init__(self, config: "ConfigDict", rngs: nnx.Rngs) -> None:
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        init_fn = nnx.initializers.normal(stddev=0.02)
        zeros = nnx.initializers.zeros_init()

        def _linear(features_in, features_out):
            return nnx.Linear(
                features_in,
                features_out,
                rngs=rngs,
                dtype=config.compute_dtype,
                kernel_init=init_fn,
                kernel_metadata={"out_sharding": (None, None)},
                bias_init=zeros,
                bias_metadata={"out_sharding": (None,)},
            )

        n = config.n_embd
        self.q_proj = _linear(n, n)
        self.k_proj = _linear(n, n)
        self.v_proj = _linear(n, n)
        self.out_proj = _linear(n, n)

    def __call__(
        self,
        inputs_q: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        inputs_q:
            Input tensor ``(batch, seq_len, n_embd)``.
        mask:
            Boolean causal mask ``(1, 1, seq_len, seq_len)``.
            ``True`` = attend, ``False`` = block (filled with ``-inf`` before
            softmax).  When ``None``, full (non-causal) attention is computed.

        Returns
        -------
        jax.Array
            Output ``(batch, seq_len, n_embd)``.
        """
        B, T, _ = inputs_q.shape

        # ── 1. Linear projections ──────────────────────────────────────────────
        q = self.q_proj(inputs_q)   # (B, T, n_embd)
        k = self.k_proj(inputs_q)
        v = self.v_proj(inputs_q)

        # ── 2. Split into heads → (B, n_head, T, head_dim) ────────────────────
        def split_heads(x: jax.Array) -> jax.Array:
            return x.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ── 3. Scaled dot-product attention ────────────────────────────────────
        # QK^T / sqrt(d_k)  →  (B, n_head, T, T)
        scale = math.sqrt(self.head_dim)
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        # Apply causal mask: blocked positions get -inf so softmax → 0
        if mask is not None:
            attn_logits = jnp.where(
                mask, attn_logits, jnp.finfo(attn_logits.dtype).min
            )

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (B, n_head, T, T)
        context = jnp.matmul(attn_weights, v)                # (B, n_head, T, head_dim)

        # ── 4. Merge heads and project ─────────────────────────────────────────
        context = context.transpose(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, n_embd)
        return self.out_proj(context)


# ── Factory ────────────────────────────────────────────────────────────────────


def build_attention_module(config: "ConfigDict", rngs: nnx.Rngs) -> nnx.Module:
    """Return the attention module specified by ``config.attention_type``.

    Parameters
    ----------
    config:
        Model config; must contain ``attention_type`` (str or
        :class:`AttentionType`) plus ``n_head``, ``n_embd``, ``compute_dtype``.
    rngs:
        NNX PRNG key bundle.

    Returns
    -------
    nnx.Module
        One of:

        * ``nnx.MultiHeadAttention`` with Flax default kernel  (``"flax"``)
        * ``nnx.MultiHeadAttention`` with flash/XLA kernel     (``"flash"``)
        * :class:`ClassicalMultiHeadAttention`                 (``"classical"``)

    Raises
    ------
    ValueError
        If ``config.attention_type`` does not match any :class:`AttentionType`.
    """
    attn_type = AttentionType(config.attention_type)
    init_fn = nnx.initializers.normal(stddev=0.02)
    zeros = nnx.initializers.zeros_init()

    # Shared kwargs for nnx.MultiHeadAttention-based types
    mha_kwargs = dict(
        num_heads=config.n_head,
        in_features=config.n_embd,
        qkv_features=config.n_embd,
        rngs=rngs,
        decode=False,
        dtype=config.compute_dtype,
        kernel_init=init_fn,
        kernel_metadata={"out_sharding": (None, None, None)},
        out_kernel_init=init_fn,
        out_kernel_metadata={"out_sharding": (None, None, None)},
        bias_init=zeros,
        bias_metadata={"out_sharding": (None,)},
        out_bias_init=zeros,
        out_bias_metadata={"out_sharding": (None,)},
    )

    if attn_type == AttentionType.FLAX:
        # nnx.MultiHeadAttention with its built-in default kernel.
        # No custom attention_fn → uses Flax's reference implementation.
        return nnx.MultiHeadAttention(**mha_kwargs)

    if attn_type == AttentionType.FLASH:
        # Custom inner kernel: cuDNN flash on GPU, XLA otherwise.
        return nnx.MultiHeadAttention(
            **mha_kwargs,
            attention_fn=functools.partial(_flash_attention_kernel, is_causal=True),
        )

    if attn_type == AttentionType.CLASSICAL:
        return ClassicalMultiHeadAttention(config, rngs)

    raise ValueError(
        f"Unknown attention_type {config.attention_type!r}. "
        f"Valid choices: {[e.value for e in AttentionType]}"
    )
