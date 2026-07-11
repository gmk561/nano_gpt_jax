"""
GPT model architecture for nano-GPT JAX.

Provides the GPT transformer model built with Flax NNX, along with
dtype utilities and compiled train/val step functions.
Attention implementations live in :mod:`attention`.

Classes
-------
MLP
    Two-layer feed-forward block with GELU activation.
Block
    Single transformer block: multi-head attention + MLP + LayerNorm.
GPT
    Full GPT language model with weight tying.

Functions
---------
is_cudnn_available()
    Returns True if cuDNN is available for flash-attention dispatch.
    See :mod:`attention`.
cast_params()
    Cast all floating-point parameters in an NNX module to a target dtype.
apply_dtype_policy()
    Apply mixed-precision param casting to attention/linear/embedding layers.
dtype_report()
    Print all float32 parameters (debugging aid).
loss_fn()
    Cross-entropy language-model loss.
train_step()
    JIT-compiled forward + backward + optimizer update.
val_step()
    JIT-compiled forward-only pass for validation.
align_acc_step()
    Convert a micro-step index to an effective gradient-accumulation step.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import optax
from flax import nnx
from ml_collections import ConfigDict

from attention import (
    build_attention_module,
    is_cudnn_available,
    MultiHeadAttention,
    MemoryEfficientAttention,
)  # noqa: F401


# ── Model modules ──────────────────────────────────────────────────────────────


class MLP(nnx.Module):
    def __init__(self, config: "ConfigDict", rngs: nnx.Rngs):
        self.config = config
        init_fn = nnx.initializers.normal(stddev=0.02)
        # Sharding: (None, None) = fully replicated for data parallelism.
        # For model parallelism later: linear_1 -> (None, 'model'), linear_2 -> ('model', None).
        self.linear_1 = nnx.Linear(
            config.n_embd,
            4 * config.n_embd,
            rngs=rngs,
            dtype=config.compute_dtype,
            kernel_init=init_fn,
            kernel_metadata={"out_sharding": (None, None)},
            bias_init=nnx.initializers.zeros_init(),
            bias_metadata={"out_sharding": (None,)},
        )
        self.linear_2 = nnx.Linear(
            4 * config.n_embd,
            config.n_embd,
            rngs=rngs,
            dtype=config.compute_dtype,
            kernel_init=init_fn,
            kernel_metadata={"out_sharding": (None, None)},
            bias_init=nnx.initializers.zeros_init(),
            bias_metadata={"out_sharding": (None,)},
        )

    def __call__(self, x: jnp.ndarray):
        return self.linear_2(nnx.gelu(self.linear_1(x)))


class RoPE(nnx.Module):

    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config

        head_dim = config.n_embd // config.n_head
        exponent = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
        freqs = 1.0 / (10000.0**exponent)

        cos = jnp.cos(jnp.outer(freqs, jnp.arange(config.max_seq_len)).T)
        sin = jnp.sin(jnp.outer(freqs, jnp.arange(config.max_seq_len)).T)

        self.cos_table = jnp.repeat(cos, 2, axis=-1)  # (max_seq_len, head_dim)
        self.sin_table = jnp.repeat(sin, 2, axis=-1)  # (max_seq_len, head_dim)

    def _rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        """Rotate pairs of features by 90°: (x1, x2) → (−x2, x1)."""
        x_perm = x.reshape(x.shape[:-1] + (-1, 2))
        rotated = jnp.stack([-x_perm[..., 1], x_perm[..., 0]], axis=-1)
        return rotated.reshape(x.shape)

    def __call__(self, x: jnp.ndarray, position_axis: int = 1):
        T = x.shape[position_axis]
        D = x.shape[-1]

        rope_cos = self.cos_table[:T, :D]
        rope_sin = self.sin_table[:T, :D]

        broadcast_shape = [1] * x.ndim
        broadcast_shape[position_axis] = T
        broadcast_shape[-1] = D

        rope_cos = jnp.reshape(rope_cos, broadcast_shape)
        rope_sin = jnp.reshape(rope_sin, broadcast_shape)

        return x * rope_cos + self._rotate_half(x) * rope_sin


class Block(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config
        # Attention module is selected by config.attention_type.
        # See attention.py / AttentionType for available options.
        self.mha = build_attention_module(
            config, rope=RoPE(config, rngs) if config.use_rope else None, rngs=rngs
        )
        self.mlp = MLP(config, rngs=rngs)
        self.layernorm_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):

        x = x + self.mha(
            self.layernorm_1(x).astype(self.config.compute_dtype), mask=mask
        )
        x = x + self.mlp(self.layernorm_2(x).astype(self.config.compute_dtype))
        return x


class GPT(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config
        # Sharding: (None, None) = replicated for data parallelism.
        # For model parallelism later: change to (None, 'model').

        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embd,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            embedding_metadata={"out_sharding": (None, None)},
            dtype=config.compute_dtype,
        )
        if not config.use_rope:
            self.wpe = nnx.Embed(
                config.max_seq_len,
                config.n_embd,
                rngs=rngs,
                embedding_init=nnx.initializers.normal(stddev=0.02),
                embedding_metadata={"out_sharding": (None, None)},
                dtype=config.compute_dtype,
            )
        self.blocks = nnx.List(
            [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        )
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)
        # Pre-compute causal mask once for the full max_seq_len; slice at call time.
        self._causal_mask = jnp.tril(
            jnp.ones((1, 1, config.max_seq_len, config.max_seq_len), dtype=jnp.bool_)
        )
        self._init_weights(rngs)

    def _init_weights(self, rngs: nnx.Rngs):
        residual_scale = 1.0 / (2 * self.config.n_layer) ** 0.5

        def _is_residual_output(module, parent, attr_name):
            """Check if this Linear is the output projection of a residual branch."""
            # MLP's second linear (projects back into residual stream)
            if isinstance(parent, MLP) and attr_name == "linear_2":
                return True
            # Attention output projection, "out", for all attention types:
            # nnx.MultiHeadAttention (FLAX / FLASH) and our custom modules
            # (CLASSICAL / MEM_EFF).
            attention_types = (
                nnx.MultiHeadAttention,
                MultiHeadAttention,
                MemoryEfficientAttention,
            )
            if isinstance(parent, attention_types) and attr_name == "out":
                return True
            return False

        def _reinit_with_sharding(variable, new_value):
            """Re-assign a variable's value while preserving its sharding metadata."""
            sharding = getattr(variable, "sharding", None)
            variable.value = new_value
            if sharding is not None:
                variable.value = jax.lax.with_sharding_constraint(
                    variable.value, sharding
                )

        def _apply(module, parent=None, _attr_name=None):
            if isinstance(module, nnx.Linear):
                stddev = (
                    0.02
                    if not _is_residual_output(module, parent, _attr_name)
                    else 0.02 * residual_scale
                )
                _reinit_with_sharding(
                    module.kernel,
                    nnx.initializers.normal(stddev=stddev)(
                        rngs.params(), module.kernel[...].shape
                    ),
                )
                if module.use_bias:
                    _reinit_with_sharding(
                        module.bias,
                        jnp.zeros(module.bias[...].shape),
                    )
            elif isinstance(module, nnx.Embed):
                _reinit_with_sharding(
                    module.embedding,
                    nnx.initializers.normal(stddev=0.02)(
                        rngs.params(), module.embedding[...].shape
                    ),
                )
            for attr_name, value in vars(module).items():
                if isinstance(value, nnx.Module):
                    _apply(value, module, attr_name)
                elif isinstance(value, (nnx.List, list)):
                    for item in value:
                        if isinstance(item, nnx.Module):
                            _apply(item, module, attr_name)
                elif isinstance(value, (nnx.Dict, dict)):
                    for item in value.values():
                        if isinstance(item, nnx.Module):
                            _apply(item, module, attr_name)

        _apply(self)

    def __call__(self, x: jax.Array):
        B, T = x.shape
        # Slice the pre-computed mask for the actual sequence length.
        mask = self._causal_mask[:, :, :T, :T]

        x = self.wte(x, out_sharding=jax.typeof(x).sharding)
        if not self.config.use_rope:
            x = x + self.wpe(jnp.arange(T))

        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x).astype(self.config.compute_dtype)
        # weight tying: reuse wte embedding matrix as output projection
        logits = x @ self.wte.embedding[...].T  # (B, T, vocab_size)
        return logits.astype(self.config.accum_dtype)


# ── Dtype utilities ────────────────────────────────────────────────────────────


def cast_params(module: nnx.Module, dtype):
    """Cast all float params in a module subtree to dtype."""

    def cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    state = nnx.state(module)
    new_state = jax.tree_util.tree_map(cast, state)
    nnx.update(module, new_state)


def apply_dtype_policy(model: nnx.Module, cfg):
    for _, module in model.iter_modules():
        if isinstance(module, (nnx.MultiHeadAttention, nnx.Linear, nnx.Embed)):
            cast_params(module, cfg.param_dtype)


def dtype_report(model: nnx.Module):
    """Print all float32 parameters — useful for verifying mixed-precision setup."""
    for path, module in nnx.iter_modules(model):
        for attr in ("kernel", "embedding", "scale", "bias"):
            param = getattr(module, attr, None)
            if param is not None and hasattr(param, "value"):
                dtype = param[...].dtype
                if dtype == jnp.float32:
                    print(f"{path} {attr} {dtype}")


# ── Loss and step functions ────────────────────────────────────────────────────


def loss_fn(model: GPT, x: jnp.ndarray, y: jnp.ndarray):
    logits = model(x)  # (B, T, vocab_size)
    B, T, V = logits.shape
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V),
        y.reshape(B * T),
    ).mean()
    return loss


@nnx.jit
def train_step(model: GPT, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
    """JIT-compiled forward + backward + optimizer update. Returns training loss."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step(model: GPT, x: jnp.ndarray, y: jnp.ndarray):
    """JIT-compiled forward-only pass for validation. Returns loss."""
    return loss_fn(model, x, y)


def align_acc_step(step: int, gradient_acc_steps: int) -> int:
    """Convert a micro-step index to an effective (post-accumulation) step."""
    return step // gradient_acc_steps
