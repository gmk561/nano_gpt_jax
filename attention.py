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

from flax.nnx.nn.linear import default_bias_init

import functools
import math
from enum import Enum

import jax
import jax.numpy as jnp
from flax import nnx
from ml_collections import ConfigDict


Array = jax.Array

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
    ``"mem_eff"``
        Chunked online-softmax (Rabe & Staats 2022) — trades peak memory for
        constant-memory attention via ``jax.lax.scan`` + ``jax.lax.map``.
    """

    FLAX = "flax"
    FLASH = "flash"
    CLASSICAL = "classical"  # reimplementation of Attention is all you need paper.
    MEM_EFF = "mem_eff"


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


# Implementation of the classical Attention is all you need paper: https://arxiv.org/abs/1706.03762
class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        use_bias: bool,
        kernel_init: nnx.initializers.Initializer = nnx.initializers.lecun_normal(),
        bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if n_embd % num_heads != 0:
            raise ValueError(
                f"Incompatible dimensions: `n_embd` ({n_embd}) must be divisible "
                f"by `num_heads` ({num_heads}) for the Flax linear layer weights to reshape correctly."
            )

        self.n_embd = n_embd
        self.use_bias = use_bias
        self.num_heads = num_heads
        head_dim = n_embd // num_heads
        self.head_dim = head_dim

        self.scale = math.sqrt(self.head_dim)

        linear = functools.partial(
            nnx.LinearGeneral,
            in_features=n_embd,
            out_features=(num_heads, head_dim),
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        self.query = linear(rngs=rngs)
        self.key = linear(rngs=rngs)
        self.value = linear(rngs=rngs)

        self.out = nnx.LinearGeneral(
            in_features=(num_heads, head_dim),
            out_features=n_embd,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
            axis=(-2, -1),
        )

    def softmax(self, qk):
        max_element = jnp.max(qk, axis=-1, keepdims=True)
        qkm = qk - max_element
        unnormalized = jnp.exp(qkm / self.scale)

        sm = unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)
        return sm

    def __call__(self, x, mask=None):
        B, T, D = x.shape

        q, k, v = (self.query(x), self.key(x), self.value(x))  # (B, T, D)
        qk = jnp.einsum("...qhd,...khd->...hqk", q, k)
        if mask is not None:
            qk = jnp.where(mask, qk, -jnp.inf)
        sm = self.softmax(qk)  # (B, H, T, T)
        att = jnp.einsum("...hqk,...khd->...qhd", sm, v)  # (B, T, H, D)

        out = self.out(att)  # (B, T, D)

        return out


class MemoryEfficientAttention(nnx.Module):
    # Implementation of https://arxiv.org/pdf/2112.05682.

    def __init__(
        self,
        n_embd,
        num_heads,
        use_bias,
        query_chunk_size: int = 64,
        key_chunk_size: int = 64,
        kernel_init: nnx.initializers.Initializer = nnx.initializers.lecun_normal(),
        bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if n_embd % num_heads != 0:
            raise ValueError(
                f"Incompatible dimensions: `n_embd` ({n_embd}) must be divisible "
                f"by `num_heads` ({num_heads}) for the Flax linear layer weights to reshape correctly."
            )

        self.n_embd = n_embd
        self.use_bias = use_bias
        self.num_heads = num_heads
        head_dim = n_embd // num_heads
        self.head_dim = head_dim
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

        self.scale = math.sqrt(self.head_dim)

        linear = functools.partial(
            nnx.LinearGeneral,
            in_features=n_embd,
            out_features=(num_heads, head_dim),
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        self.query = linear(rngs=rngs)
        self.key = linear(rngs=rngs)
        self.value = linear(rngs=rngs)

        self.out = nnx.LinearGeneral(
            in_features=(num_heads, head_dim),
            out_features=n_embd,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
            axis=(-2, -1),
        )

    def _chunk_attention(self, query, keys, values, B, H, D, T, mask=None):
        """Scan over all key/value chunks for one query chunk and return the
        normalised attention output.

        Parameters
        ----------
        query_chunk : (B, chunk_size, H, D)
        k, v        : (B, T, H, D)
        B, H, D, T  : int — static dimension sizes
        """

        @functools.partial(jax.checkpoint, prevent_cse=False)
        def chunk_scanner(idx):
            sliced_keys = jax.lax.dynamic_slice(
                keys,
                start_indices=(0, idx * self.key_chunk_size, 0, 0),
                slice_sizes=(B, self.key_chunk_size, H, D),
            )  # (B, C, H, D)
            sliced_values = jax.lax.dynamic_slice(
                values,
                start_indices=(0, idx * self.key_chunk_size, 0, 0),
                slice_sizes=(B, self.key_chunk_size, H, D),
            )  # (B, C, H, D)

            attention_weights = jnp.einsum(
                "bchd,bkhd->bhck",
                query,
                sliced_keys,
            )  # (B, H, C, C)
            attention_weights = attention_weights / self.scale  # (B, H, C, C)

            if mask is not None:
                sliced_mask = jax.lax.dynamic_slice(
                    mask,
                    start_indices=(0, 0, 0, idx * self.key_chunk_size),
                    slice_sizes=(1, 1, self.query_chunk_size, self.key_chunk_size),
                )
                attention_weights = jnp.where(sliced_mask, attention_weights, -jnp.inf)

            max_att_weight = jnp.max(
                attention_weights, axis=-1, keepdims=True
            )  # (B, H, C, 1)

            max_att_weight = jnp.maximum(max_att_weight, -1e9)
            attention_weights = attention_weights - max_att_weight
            # Clamp it so it can never drop below a safe finite value
            # As -inf - -inf = NaN

            exp_att_weights = jnp.exp(attention_weights)  # (B, H, C, C)
            exp_att_values = jnp.einsum(
                "bhqk,bkhd->bhqd", exp_att_weights, sliced_values
            )  # (B, C, H, D)

            return max_att_weight, jnp.sum(exp_att_weights, axis=-1), exp_att_values

        max_att_weights, exp_attention_weights, exp_att_values = jax.lax.map(
            chunk_scanner, jnp.arange(math.ceil(T / self.key_chunk_size))
        )

        global_max_att_weight = jnp.max(max_att_weights, axis=0)
        exp_max_att_diff = jnp.exp(
            max_att_weights - global_max_att_weight
        )  # (N_K, B, H, C, 1)

        exp_att_values = jnp.sum(
            exp_att_values * exp_max_att_diff, axis=0, keepdims=False
        )  # (B,T,H,D)

        exp_attention_weights = jnp.sum(
            exp_attention_weights * exp_max_att_diff[..., 0], axis=0, keepdims=False
        )  # (B,T,H,D)

        return exp_att_values / exp_attention_weights[..., None]

    def __call__(self, x, mask=None):
        q, k, v = self.query(x), self.key(x), self.value(x)

        B, T, H, D = q.shape

        def _query_chunk_processor(idx: int, _):
            query_chunk = jax.lax.dynamic_slice(
                q,
                start_indices=(0, idx * self.query_chunk_size, 0, 0),
                slice_sizes=(B, self.query_chunk_size, H, D),
            )

            mask_chunk = None
            if mask is not None:
                mask_chunk = jax.lax.dynamic_slice(
                    mask,
                    start_indices=(0, 0, idx * self.query_chunk_size, 0),
                    slice_sizes=(1, 1, self.query_chunk_size, T),
                )

            return idx + 1, self._chunk_attention(
                query_chunk, k, v, B, H, D, T, mask=mask_chunk
            )

        num_chunks = int(math.ceil(T / self.query_chunk_size))
        _, att = jax.lax.scan(
            _query_chunk_processor, init=0, xs=None, length=num_chunks
        )  # (num_chunks, B, C, H, D)

        return self.out(att.transpose(1, 0, 3, 2, 4).reshape(B, T, H, D))


def build_attention_module(config: ConfigDict, rngs: nnx.Rngs) -> nnx.Module:
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
        return MultiHeadAttention(
            n_embd=config.n_embd,
            num_heads=config.n_head,
            use_bias=config.use_attention_bias,
            kernel_init=init_fn,
            bias_init=zeros,
            rngs=rngs,
        )

    if attn_type == AttentionType.MEM_EFF:
        # Chunk sizes default to 64; can be overridden via config.
        query_chunk_size = getattr(config, "query_chunk_size", 64)
        key_chunk_size = getattr(config, "key_chunk_size", 64)
        return MemoryEfficientAttention(
            n_embd=config.n_embd,
            num_heads=config.n_head,
            use_bias=config.use_attention_bias,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            kernel_init=init_fn,
            bias_init=zeros,
            rngs=rngs,
        )

    raise ValueError(
        f"Unknown attention_type {config.attention_type!r}. "
        f"Valid choices: {[e.value for e in AttentionType]}"
    )
