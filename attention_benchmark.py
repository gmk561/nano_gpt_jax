"""
attention_benchmark.py
======================

Benchmark memory, FLOPs and wall-clock speed for every attention
implementation in attention.py.

Implementations under test
---------------------------
* MultiHeadAttention     – classical explicit SDPA (Vaswani et al.)
* MemoryEfficientAttention – chunked online-softmax (Rabe & Staats 2022)
* nnx.MultiHeadAttention (flax)  – Flax reference kernel
* nnx.MultiHeadAttention (flash) – JAX dot_product_attention / cuDNN

Usage
------
    python attention_benchmark.py                    # default config
    python attention_benchmark.py --seq 512 --reps 50
"""

from __future__ import annotations

import argparse
import functools
import math
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from attention import (
    MemoryEfficientAttention,
    MultiHeadAttention,
    _flash_attention_kernel,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_MESH = jax.make_mesh((1, 1), ("data", "model"))


def _make_module(cls, n_embd, n_head, seed=0, **kwargs):
    with jax.set_mesh(_MESH):
        nnx.use_eager_sharding(True)
        return cls(
            n_embd=n_embd,
            num_heads=n_head,
            use_bias=False,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros_init(),
            rngs=nnx.Rngs(seed),
            **kwargs,
        )


def _make_flax_module(n_embd, n_head, seed=0, attention_fn=None):
    kwargs = dict(
        num_heads=n_head,
        in_features=n_embd,
        qkv_features=n_embd,
        rngs=nnx.Rngs(seed),
        decode=False,
        dtype=jnp.float32,
        kernel_init=nnx.initializers.normal(stddev=0.02),
        out_kernel_init=nnx.initializers.normal(stddev=0.02),
        bias_init=nnx.initializers.zeros_init(),
        out_bias_init=nnx.initializers.zeros_init(),
    )
    if attention_fn is not None:
        kwargs["attention_fn"] = attention_fn
    with jax.set_mesh(_MESH):
        nnx.use_eager_sharding(True)
        return nnx.MultiHeadAttention(**kwargs)


def _jit_forward(module: nnx.Module) -> Callable:
    """Return a jit-compiled forward function for *module*."""
    graphdef, state = nnx.split(module)

    @jax.jit
    def forward(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)

    return functools.partial(forward, state)


# ── Memory analysis ────────────────────────────────────────────────────────────


@dataclass
class MemStats:
    temp_bytes: int
    arg_bytes: int
    out_bytes: int
    alias_bytes: int

    @property
    def peak_bytes(self) -> int:
        return self.temp_bytes + self.arg_bytes + self.out_bytes - self.alias_bytes


def _memory_analysis(module: nnx.Module, x_shape: tuple) -> MemStats:
    graphdef, state = nnx.split(module)

    state_abstract = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state
    )
    x_abstract = jax.ShapeDtypeStruct(x_shape, jnp.float32)

    @jax.jit
    def forward(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)

    a = forward.lower(state_abstract, x_abstract).compile().memory_analysis()
    return MemStats(
        temp_bytes=a.temp_size_in_bytes,
        arg_bytes=a.argument_size_in_bytes,
        out_bytes=a.output_size_in_bytes,
        alias_bytes=a.alias_size_in_bytes,
    )


# ── FLOPs analysis ─────────────────────────────────────────────────────────────


@dataclass
class FlopsStats:
    flops: int
    """Total FLOPs reported by XLA cost_analysis (sum across all HLO ops)."""


def _flop_analysis(module: nnx.Module, x_shape: tuple) -> FlopsStats:
    graphdef, state = nnx.split(module)

    state_abstract = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state
    )
    x_abstract = jax.ShapeDtypeStruct(x_shape, jnp.float32)

    @jax.jit
    def forward(state, x):
        m = nnx.merge(graphdef, state)
        return m(x)

    lowered = forward.lower(state_abstract, x_abstract)
    analyses = lowered.cost_analysis()

    total = 0
    if analyses:
        for entry in analyses:
            if isinstance(entry, dict):
                # JAX >= 0.4.x: list of dicts
                total += entry.get("flops", 0.0)
            elif isinstance(entry, str):
                # Older JAX: list of "key: value\n..." strings
                for line in entry.splitlines():
                    if line.strip().lower().startswith("flops"):
                        try:
                            total += float(line.split(":")[-1].strip())
                        except ValueError:
                            pass

    return FlopsStats(flops=int(total))


# ── Speed benchmark ────────────────────────────────────────────────────────────


@dataclass
class SpeedStats:
    warmup_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def _speed_benchmark(
    module: nnx.Module, x: jax.Array, reps: int = 100
) -> SpeedStats:
    import statistics

    fwd = _jit_forward(module)

    # Two warmup calls to ensure compilation and buffer allocation.
    fwd(x).block_until_ready()
    t0 = time.perf_counter()
    fwd(x).block_until_ready()
    warmup_ms = (time.perf_counter() - t0) * 1e3

    times_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fwd(x).block_until_ready()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    return SpeedStats(
        warmup_ms=warmup_ms,
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
    )


# ── Formatting ─────────────────────────────────────────────────────────────────


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:,} {unit}"
        n //= 1024
    return f"{n:,} TiB"


def _fmt_flops(n: int) -> str:
    for unit, div in (("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:,} FLOPs"


def _print_row(name: str, mem: MemStats | None, flops: FlopsStats | None, speed: SpeedStats | None):
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    if mem is not None:
        print(f"  Memory (XLA static buffer estimates)")
        print(f"    temp (scratch):  {_fmt_bytes(mem.temp_bytes)}")
        print(f"    args (params+x): {_fmt_bytes(mem.arg_bytes)}")
        print(f"    output:          {_fmt_bytes(mem.out_bytes)}")
        print(f"    peak estimate:   {_fmt_bytes(mem.peak_bytes)}")
    if flops is not None:
        print(f"  FLOPs (XLA cost_analysis): {_fmt_flops(flops.flops)}")
    if speed is not None:
        print(f"  Wall-clock speed ({speed.mean_ms:.2f} ± {speed.std_ms:.2f} ms per call)")
        print(f"    warmup: {speed.warmup_ms:.2f} ms")
        print(f"    mean:   {speed.mean_ms:.2f} ms")
        print(f"    min:    {speed.min_ms:.2f} ms")
        print(f"    max:    {speed.max_ms:.2f} ms")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Attention implementation benchmark")
    parser.add_argument("--batch",    type=int, default=2,   help="Batch size")
    parser.add_argument("--seq",      type=int, default=64,  help="Sequence length T")
    parser.add_argument("--embd",     type=int, default=64,  help="Embedding dim n_embd")
    parser.add_argument("--heads",    type=int, default=4,   help="Number of heads")
    parser.add_argument("--reps",     type=int, default=50,  help="Timing repetitions")
    parser.add_argument("--no-mem",   action="store_true",   help="Skip memory analysis")
    parser.add_argument("--no-flops", action="store_true",   help="Skip FLOPs analysis")
    parser.add_argument("--no-speed", action="store_true",   help="Skip speed benchmark")
    parser.add_argument(
        "--impl", nargs="+",
        choices=["classical", "mem_eff", "flax", "flash"],
        default=["classical", "mem_eff", "flax", "flash"],
        help="Which implementations to benchmark",
    )
    args = parser.parse_args()

    B, T, E, H = args.batch, args.seq, args.embd, args.heads
    x_shape = (B, T, E)
    x = jax.random.normal(jax.random.PRNGKey(0), x_shape)

    print(f"\n{'═' * 70}")
    print(f"  Attention Benchmark")
    print(f"  B={B}  T={T}  n_embd={E}  n_head={H}  head_dim={E // H}")
    print(f"  reps={args.reps}  device={jax.default_backend()}")
    print(f"{'═' * 70}")

    # ── Build modules ──────────────────────────────────────────────────────────
    modules: dict[str, nnx.Module] = {}

    if "classical" in args.impl:
        modules["MultiHeadAttention (classical)"] = _make_module(MultiHeadAttention, E, H)

    if "mem_eff" in args.impl:
        modules["MemoryEfficientAttention (chunked)"] = _make_module(
            MemoryEfficientAttention, E, H,
            query_chunk_size=min(T, 16),
            key_chunk_size=min(T, 16),
        )

    if "flax" in args.impl:
        modules["nnx.MultiHeadAttention (flax default)"] = _make_flax_module(E, H)

    if "flash" in args.impl:
        modules["nnx.MultiHeadAttention (flash/XLA)"] = _make_flax_module(
            E, H,
            attention_fn=functools.partial(_flash_attention_kernel, is_causal=True),
        )

    # ── Run benchmarks ─────────────────────────────────────────────────────────
    for name, module in modules.items():
        mem = flops = speed = None

        if not args.no_mem:
            try:
                mem = _memory_analysis(module, x_shape)
            except Exception as e:
                print(f"  [mem ERROR] {name}: {e}")

        if not args.no_flops:
            try:
                flops = _flop_analysis(module, x_shape)
            except Exception as e:
                print(f"  [flops ERROR] {name}: {e}")

        if not args.no_speed:
            try:
                speed = _speed_benchmark(module, x, reps=args.reps)
            except Exception as e:
                print(f"  [speed ERROR] {name}: {e}")

        _print_row(name, mem, flops, speed)

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
