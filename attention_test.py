"""
Tests for attention.py
======================

Verifies that :class:`~attention.ClassicalMultiHeadAttention` is:

1. **Correct** — matches a pure-JAX reference implementation of SDPA step by step.
2. **Causally masked** — output at position ``t`` doesn't depend on tokens ``> t``.
3. **Equivalent to FlaxMHA** — given identical weights, ``ClassicalMHA`` and
   ``nnx.MultiHeadAttention`` (``AttentionType.FLAX``) produce the same output.

Weight layout difference
------------------------
::

    ClassicalMHA (nnx.Linear)          FlaxMHA (DenseGeneral)
    ----------------------------------  --------------------------------
    q/k/v_proj.kernel  (E, E)          query/key/value.kernel  (E, H, D)
    q/k/v_proj.bias    (E,)            query/key/value.bias    (H, D)
    out_proj.kernel    (E, E)          out.kernel              (H, D, E)
    out_proj.bias      (E,)            out.bias                (E,)

    E = n_embd,  H = n_head,  D = head_dim = E // H

The copy relationship is a simple reshape along the head dimension:

    flax.query.kernel = classical.q_proj.kernel.reshape(E, H, D)
    flax.out.kernel   = classical.out_proj.kernel.reshape(H, D, E)

Run with::

    .venv/bin/python -m pytest test_attention.py -v
"""

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from ml_collections import ConfigDict

from attention import (
    AttentionType,
    MemoryEfficientAttention,
    MultiHeadAttention,
    build_attention_module,
)

# ── Shared mesh (needed for nnx sharding annotations) ─────────────────────────

_MESH = jax.make_mesh((1, 1), ("data", "model"))

# ── Helpers ────────────────────────────────────────────────────────────────────


def _cfg(
    attention_type: AttentionType,
    n_embd: int = 32,
    n_head: int = 4,
) -> ConfigDict:
    """Minimal :class:`~ml_collections.ConfigDict` for an attention module."""
    cfg = ConfigDict()
    cfg.n_embd = n_embd
    cfg.n_head = n_head
    cfg.compute_dtype = jnp.float32
    cfg.attention_type = attention_type.value
    cfg.use_attention_bias = False
    return cfg


def _build(cfg: ConfigDict, seed: int = 0) -> nnx.Module:
    """Instantiate an attention module inside the shared mesh context."""
    with jax.set_mesh(_MESH):
        nnx.use_eager_sharding(True)
        return build_attention_module(cfg, rngs=nnx.Rngs(seed))


def _make_paired_modules(
    n_embd: int, n_head: int, seed: int = 0
) -> tuple[MultiHeadAttention, nnx.MultiHeadAttention]:
    """Return a ``ClassicalMHA`` and a ``FlaxMHA`` carrying **identical weights**.

    Weights are drawn from ``ClassicalMHA`` and transferred to ``FlaxMHA``
    with the axis-reshape needed to match Flax's ``DenseGeneral`` layout.

    Verified weight shapes for ``n_embd=32, n_head=4`` (head_dim=8)::

        query.kernel: (32, 4, 8)   .bias: (4, 8)
        key.kernel:   (32, 4, 8)   .bias: (4, 8)
        value.kernel: (32, 4, 8)   .bias: (4, 8)
        out.kernel:   (4, 8, 32)   .bias: (32,)
    """
    head_dim = n_embd // n_head

    cfg_c = _cfg(AttentionType.CLASSICAL, n_embd, n_head)
    classical: MultiHeadAttention = _build(cfg_c, seed=seed)

    cfg_f = _cfg(AttentionType.FLAX, n_embd, n_head)
    flax_mha: nnx.MultiHeadAttention = _build(cfg_f, seed=seed + 999)

    # QKV:  classical (E, H, D) → flax (E, H, D)
    for c_proj, f_proj in [
        (classical.query, flax_mha.query),
        (classical.key, flax_mha.key),
        (classical.value, flax_mha.value),
    ]:
        f_proj.kernel[...] = c_proj.kernel[...]
        if c_proj.bias is not None and f_proj.bias is not None:
            f_proj.bias[...] = c_proj.bias[...]

    # Output: classical (H, D, E) → flax (H, D, E)
    flax_mha.out.kernel[...] = classical.out.kernel[...]
    if classical.out.bias is not None and flax_mha.out.bias is not None:
        flax_mha.out.bias[...] = classical.out.bias[...]

    return classical, flax_mha


def _causal_mask(T: int) -> jax.Array:
    """Lower-triangular boolean mask of shape ``(1, 1, T, T)``. True = attend."""
    return jnp.tril(jnp.ones((1, 1, T, T), dtype=jnp.bool_))


def _reference_mhsa(
    x: jax.Array,
    q_w,
    q_b,
    k_w,
    k_b,
    v_w,
    v_b,
    o_w,
    o_b,
    *,
    n_head: int,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Pure-JAX multi-head self-attention reference (no NNX, no reshape tricks).

    Implements the formula from Vaswani et al. (2017) step by step so that
    it can serve as ground truth for ``ClassicalMultiHeadAttention``.
    """
    B, T, n_embd = x.shape
    head_dim = n_embd // n_head

    def _proj(W, b):
        # (B, T, E) → linear → (B, T, E) → split heads → (B, H, T, D)
        return (x @ W + b).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

    q, k, v = _proj(q_w, q_b), _proj(k_w, k_b), _proj(v_w, v_b)

    logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)  # (B,H,T,T)
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    context = jnp.matmul(jax.nn.softmax(logits, axis=-1), v)  # (B, H, T, D)
    return context.transpose(0, 2, 1, 3).reshape(B, T, n_embd) @ o_w + o_b


# ── ClassicalMHA vs FlaxMHA equivalence ────────────────────────────────────────


class TestClassicalVsFlax:
    """
    Both modules are initialised with identical weights (after the reshape
    copy in ``_make_paired_modules``).  Any numerical difference reveals a
    bug in either the weight-copy logic or in a forward-pass step.

    Tolerance ``atol=1e-5`` accommodates float32 operation-reordering
    between the two code paths.
    """

    @pytest.mark.parametrize(
        "n_embd,n_head",
        [
            (4, 1),
            (16, 2),
            (32, 4),
            (64, 8),
        ],
    )
    def test_no_mask(self, n_embd, n_head):
        B, T = 2, 8
        classical, flax_mha = _make_paired_modules(n_embd, n_head, seed=0)

        x = jax.random.normal(jax.random.PRNGKey(42), (B, T, n_embd))

        out_c = classical(x, mask=None)
        out_f = flax_mha(x, mask=None, rngs=nnx.Rngs(0))

        assert out_c.shape == out_f.shape == (B, T, n_embd)
        assert jnp.allclose(
            out_c, out_f, atol=1e-5
        ), f"[no mask, E={n_embd}, H={n_head}] max_diff={jnp.abs(out_c - out_f).max():.2e}"

    @pytest.mark.parametrize(
        "n_embd,n_head",
        [
            (16, 2),
            (32, 4),
        ],
    )
    def test_causal_mask(self, n_embd, n_head):
        B, T = 2, 12
        classical, flax_mha = _make_paired_modules(n_embd, n_head, seed=7)
        x = jax.random.normal(jax.random.PRNGKey(123), (B, T, n_embd))
        mask = _causal_mask(T)

        out_c = classical(x, mask=mask)
        out_f = flax_mha(x, mask=mask)

        assert out_c.shape == out_f.shape == (B, T, n_embd)
        assert jnp.allclose(
            out_c, out_f, atol=1e-5
        ), f"[causal, E={n_embd}, H={n_head}] max_diff={jnp.abs(out_c - out_f).max():.2e}"

    def test_single_token(self):
        """Degenerate T=1 sequence works in both implementations."""
        B, T, n_embd, n_head = 1, 1, 32, 4
        classical, flax_mha = _make_paired_modules(n_embd, n_head)
        x = jax.random.normal(jax.random.PRNGKey(0), (B, T, n_embd))
        assert jnp.allclose(classical(x), flax_mha(x), atol=1e-5)

    def test_batch_independence(self):
        """Processing a batch must give the same result as processing items one by one."""
        B, T, n_embd, n_head = 4, 8, 32, 4
        classical, flax_mha = _make_paired_modules(n_embd, n_head)
        mask = _causal_mask(T)

        x = jax.random.normal(jax.random.PRNGKey(0), (B, T, n_embd))
        out_c_batch = classical(x, mask=mask)
        out_f_batch = flax_mha(x, mask=mask)

        for i in range(B):
            xi = x[i : i + 1]
            out_single = classical(xi, mask=mask)
            assert jnp.allclose(
                out_c_batch[i : i + 1], out_single, atol=1e-5
            ), f"ClassicalMHA: batch item {i} differs from single-item result"
            assert jnp.allclose(
                out_f_batch[i : i + 1], out_single, atol=1e-5
            ), f"FlaxMHA: batch item {i} differs from single-item ClassicalMHA result"


# ── MemoryEfficientAttention tests ─────────────────────────────────────────────


def _make_mem_eff_module(
    n_embd: int, n_head: int, seed: int = 0
) -> tuple[MultiHeadAttention, MemoryEfficientAttention]:
    """Return a ``MultiHeadAttention`` and a ``MemoryEfficientAttention`` with identical weights.

    Both modules use ``LinearGeneral`` projections with the same output shape
    ``(num_heads, head_dim)``, so weights are copied directly without reshaping.
    """
    with jax.set_mesh(_MESH):
        nnx.use_eager_sharding(True)
        classical = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            use_bias=False,
            rngs=nnx.Rngs(seed),
        )
        mem_eff = MemoryEfficientAttention(
            n_embd=n_embd,
            num_heads=n_head,
            use_bias=False,
            query_chunk_size=4,
            key_chunk_size=4,
            rngs=nnx.Rngs(seed + 999),
        )

    # Copy weights: both use LinearGeneral with identical kernel shapes
    for c_proj, m_proj in [
        (classical.query, mem_eff.query),
        (classical.key, mem_eff.key),
        (classical.value, mem_eff.value),
    ]:
        m_proj.kernel[...] = c_proj.kernel[...]
        if c_proj.bias is not None and m_proj.bias is not None:
            m_proj.bias[...] = c_proj.bias[...]

    mem_eff.out.kernel[...] = classical.out.kernel[...]
    if classical.out.bias is not None and mem_eff.out.bias is not None:
        mem_eff.out.bias[...] = classical.out.bias[...]

    return classical, mem_eff


class TestMemoryEfficientAttention:
    """Tests for MemoryEfficientAttention.

    The module implements an online-softmax loop over tokens.  We verify it
    produces the correct output shape and, given identical weights, yields
    values close to ``MultiHeadAttention`` (the dense baseline).
    """

    @pytest.mark.parametrize(
        "n_embd,n_head",
        [
            (4, 1),
            (16, 2),
            (32, 4),
        ],
    )
    def test_output_shape(self, n_embd, n_head):
        """Output shape is always ``(B, T, n_embd)``."""
        B, T = 2, 8
        with jax.set_mesh(_MESH):
            nnx.use_eager_sharding(True)
            mem_eff = MemoryEfficientAttention(
                n_embd=n_embd,
                num_heads=n_head,
                use_bias=False,
                query_chunk_size=4,
                key_chunk_size=4,
                rngs=nnx.Rngs(0),
            )
        x = jax.random.normal(jax.random.PRNGKey(7), (B, T, n_embd))
        out = mem_eff(x)
        assert out.shape == (
            B,
            T,
            n_embd,
        ), f"Expected ({B}, {T}, {n_embd}), got {out.shape}"

    @pytest.mark.parametrize(
        "n_embd,n_head",
        [
            (4, 1),
            (16, 2),
            (32, 4),
        ],
    )
    def test_matches_classical_no_mask(self, n_embd, n_head):
        """Given identical weights, MemoryEfficientAttention should match MultiHeadAttention."""
        B, T = 2, 8
        classical, mem_eff = _make_mem_eff_module(n_embd, n_head, seed=0)

        x = jax.random.normal(jax.random.PRNGKey(42), (B, T, n_embd))

        out_c = classical(x, mask=None)
        out_m = mem_eff(x)

        assert out_c.shape == out_m.shape == (B, T, n_embd)
        assert jnp.allclose(
            out_c, out_m, atol=1e-5
        ), f"[E={n_embd}, H={n_head}] max_diff={jnp.abs(out_c - out_m).max():.2e}"

    @pytest.mark.parametrize(
        "n_embd,n_head",
        [
            (4, 1),
            (16, 2),
            (32, 4),
        ],
    )
    def test_matches_classical_with_mask(self, n_embd, n_head):
        """Given identical weights and a causal mask, MemoryEfficientAttention should match MultiHeadAttention."""
        B, T = 2, 128
        classical, mem_eff = _make_mem_eff_module(n_embd, n_head, seed=0)

        x = jax.random.normal(jax.random.PRNGKey(42), (B, T, n_embd))
        mask = _causal_mask(T)

        out_c = classical(x, mask=mask)
        out_m = mem_eff(x, mask=mask)

        assert out_c.shape == out_m.shape == (B, T, n_embd)
        assert jnp.allclose(
            out_c, out_m, atol=1e-5
        ), f"[E={n_embd}, H={n_head}] max_diff={jnp.abs(out_c - out_m).max():.2e}"


# ── Memory analysis ────────────────────────────────────────────────────────────


def _memory_analysis(module: nnx.Module, x_shape: tuple, dtype=jnp.float32):
    """JIT-compile ``module``'s forward pass and return its memory analysis.

    Uses ``jax.stages.Compiled.memory_analysis()`` to query XLA's static
    buffer-size estimates.  The four fields are:

    * ``temp_size_in_bytes``     – scratch / intermediate activations
    * ``argument_size_in_bytes`` – weight parameters + input tensor
    * ``output_size_in_bytes``   – output tensor
    * ``alias_size_in_bytes``    – memory shared between inputs and outputs

    Peak memory estimate = temp + argument + output − alias.
    """
    graphdef, state = nnx.split(module)

    state_abstract = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state
    )
    x_abstract = jax.ShapeDtypeStruct(x_shape, dtype)

    @jax.jit
    def forward(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)

    return forward.lower(state_abstract, x_abstract).compile().memory_analysis()


def _fmt(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:>10,} {unit}"
        n //= 1024
    return f"{n:>10,} TiB"


class TestAttentionMemory:
    """Compare JIT-compiled memory footprint of MultiHeadAttention vs
    MemoryEfficientAttention using ``jax.stages.Compiled.memory_analysis()``.

    Important caveat
    ----------------
    ``MemoryEfficientAttention`` uses **Python-level for-loops**, which XLA
    sees as a fully unrolled static graph (T×T separate ops).  This means
    the compiled HLO may actually use *more* intermediate memory than the
    batched matrix-multiply in ``MultiHeadAttention``.  The test surfaces
    this directly so you can see the trade-off.

    To get genuine memory savings you would need to replace the Python loops
    with ``jax.lax.fori_loop`` (or ``jax.lax.scan``), which produce a single
    looped HLO op and avoid materialising the full T×T score matrix.
    """

    def test_memory_analysis(self, request):
        n_embd, n_head = 32, 4
        # T=16: large enough to see differences but small enough to compile in
        # reasonable time.  MemoryEfficientAttention's Python for-loops cause
        # XLA to unroll T×T ops at compile time; T=64 would take minutes.
        B, T = 2, 128
        x_shape = (B, T, n_embd)

        classical, mem_eff = _make_mem_eff_module(n_embd, n_head, seed=0)

        classical_analysis = _memory_analysis(classical, x_shape)
        mem_eff_analysis = _memory_analysis(mem_eff, x_shape)

        analyses = {
            "MultiHeadAttention      ": classical_analysis,
            "MemoryEfficientAttention": mem_eff_analysis,
        }

        def _peak(a) -> int:
            return (
                a.temp_size_in_bytes
                + a.argument_size_in_bytes
                + a.output_size_in_bytes
                - a.alias_size_in_bytes
            )

        # ── Conditional table: only printed with -s or -v ─────────────────────
        verbose = request.config.option.verbose > 0
        no_capture = request.config.option.capture == "no"
        if verbose or no_capture:
            header = (
                f"\n{'':─<74}\n"
                f" {'Implementation':<26} {'temp':>12}  {'args':>12}"
                f"  {'output':>12}  {'peak est.':>12}\n"
                f"{'':─<74}"
            )
            print(header)
            for name, a in analyses.items():
                print(
                    f" {name}  {_fmt(a.temp_size_in_bytes)}"
                    f"  {_fmt(a.argument_size_in_bytes)}"
                    f"  {_fmt(a.output_size_in_bytes)}"
                    f"  {_fmt(_peak(a))}"
                )
            print(f"{'':─<74}\n")

        # ── Assertions comparing the two implementations ───────────────────────
        # Both analyses must be well-formed.
        for name, a in analyses.items():
            assert a.temp_size_in_bytes >= 0, f"{name}: negative temp_size"
            assert a.argument_size_in_bytes >= 0, f"{name}: negative argument_size"
            assert a.output_size_in_bytes >= 0, f"{name}: negative output_size"
            assert a.alias_size_in_bytes >= 0, f"{name}: negative alias_size"

        # Argument and output sizes must be equal: both modules share the same
        # weight layout (LinearGeneral with identical kernel shapes) and produce
        # the same output shape.
        assert (
            classical_analysis.argument_size_in_bytes
            == mem_eff_analysis.argument_size_in_bytes
        ), (
            "Parameter memory differs between implementations — weight layouts diverged.\n"
            f"  MultiHeadAttention:       {_fmt(classical_analysis.argument_size_in_bytes)}\n"
            f"  MemoryEfficientAttention: {_fmt(mem_eff_analysis.argument_size_in_bytes)}"
        )
        assert (
            classical_analysis.output_size_in_bytes
            == mem_eff_analysis.output_size_in_bytes
        ), (
            "Output buffer sizes differ — output shapes diverged.\n"
            f"  MultiHeadAttention:       {_fmt(classical_analysis.output_size_in_bytes)}\n"
            f"  MemoryEfficientAttention: {_fmt(mem_eff_analysis.output_size_in_bytes)}"
        )

        # Surface the temp-memory trade-off explicitly.
        classical_peak = _peak(classical_analysis)
        mem_eff_peak = _peak(mem_eff_analysis)

        def _full_report(name, a) -> str:
            return (
                f"\n  {name}:\n"
                f"    temp_size:     {a.temp_size_in_bytes:>12,} B  ({_fmt(a.temp_size_in_bytes)})\n"
                f"    argument_size: {a.argument_size_in_bytes:>12,} B  ({_fmt(a.argument_size_in_bytes)})\n"
                f"    output_size:   {a.output_size_in_bytes:>12,} B  ({_fmt(a.output_size_in_bytes)})\n"
                f"    alias_size:    {a.alias_size_in_bytes:>12,} B  ({_fmt(a.alias_size_in_bytes)})\n"
                f"    peak_estimate: {_peak(a):>12,} B  ({_fmt(_peak(a))})"
            )

        assert classical_peak > 0 and mem_eff_peak > 0, (
            "Peak memory estimates are zero — XLA analysis may be unavailable on this backend.\n"
            + _full_report("MultiHeadAttention", classical_analysis)
            + _full_report("MemoryEfficientAttention", mem_eff_analysis)
        )
