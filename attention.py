"""
Attention mechanisms for nano-GPT JAX.

Provides two multi-head attention backends selectable via
``cfg.model.attention_type`` in ``config.py``:

``"flash"``
    ``nnx.MultiHeadAttention`` with ``jax.nn.dot_product_attention`` as the
    inner kernel.  Dispatches to cuDNN flash attention on supported GPUs and
    falls back to the XLA implementation otherwise.  This is the production
    default for GPU training.

``"flax"``
    ``nnx.MultiHeadAttention`` with its default built-in attention kernel.
    Numerically stable reference baseline, used for CPU testing.

Configure via ``cfg.model.attention_type`` (``"flash"`` or ``"flax"``).
"""

from __future__ import annotations

from flax.nnx.nn.attention import dot_product_attention as flax_dot_product_attention

import functools

import jax
from flax import nnx
from ml_collections import ConfigDict


def is_cudnn_available() -> bool:
    """Return ``True`` if a cuDNN runtime is accessible via JAX."""
    try:
        from jax._src.lib import cuda_versions

        return (
            cuda_versions is not None and cuda_versions.cudnn_get_version() is not None
        )
    except (ImportError, AttributeError, RuntimeError):
        return False


def _flash_attention_kernel(
    query, key, value, rope=None, bias=None, mask=None, is_causal: bool = True, **kwargs
):
    """JAX ``dot_product_attention`` dispatching to cuDNN when available.

    This function is passed as ``attention_fn`` to ``nnx.MultiHeadAttention``
    and handles only the core QK^T V computation — the QKV/output projections
    are still managed by ``nnx.MultiHeadAttention``.
    """
    impl = "cudnn" if is_cudnn_available() else "xla"

    query = query if rope is None else rope(query, position_axis=2)
    key = key if rope is None else rope(key, position_axis=2)

    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask if not is_causal else None,
        is_causal=is_causal,
        implementation=impl,
    )


def build_attention_module(
    config: ConfigDict,
    *,
    rope: nnx.Module | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Build a multi-head attention module from config.

    Parameters
    ----------
    config:
        Model config containing ``attention_type``, ``n_head``, ``n_embd``,
        and ``compute_dtype``.
    rope:
        Optional RoPE module for rotary position embeddings.
    rngs:
        Flax NNX random number generators.

    Returns
    -------
    An ``nnx.MultiHeadAttention`` instance.

    Raises
    ------
    ValueError
        If ``config.attention_type`` is not ``"flash"`` or ``"flax"``.
    """
    attn_type = config.attention_type
    init_fn = nnx.initializers.normal(stddev=0.02)
    zeros = nnx.initializers.zeros_init()

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
        normalize_qk=True,
    )

    if attn_type == "flax":
        if rope is not None:

            def rope_flax_attention_fn(
                query, key, value, bias=None, mask=None, **kwargs
            ):
                query = rope(query, position_axis=2)
                key = rope(key, position_axis=2)
                return flax_dot_product_attention(
                    query, key, value, bias=bias, mask=mask, **kwargs
                )

            mha_kwargs["attention_fn"] = rope_flax_attention_fn
        return nnx.MultiHeadAttention(**mha_kwargs)

    if attn_type == "flash":
        return nnx.MultiHeadAttention(
            **mha_kwargs,
            attention_fn=functools.partial(
                _flash_attention_kernel, rope=rope, is_causal=True
            ),
        )

    raise ValueError(
        f"Unknown attention_type {attn_type!r}. Valid choices: 'flash', 'flax'."
    )


# All concrete attention module classes — used for isinstance checks elsewhere
# (e.g. weight initialisation in model.py).
ATTENTION_TYPES = (nnx.MultiHeadAttention,)
