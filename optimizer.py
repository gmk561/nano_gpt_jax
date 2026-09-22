"""
Optimizer utilities for nano-GPT JAX.

Provides:
- AdamW and Muon optimizer builders.
- Learning rate schedule construction (cosine & trapezoidal).
- Weight decay masking and Muon parameter classification.
- High-level :func:`build_optimizer` entry point used by ``train.py``.

Select the optimizer via ``cfg.optimizer.type`` (``"adamw"`` or ``"muon"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import optax
from flax import nnx
from ml_collections import ConfigDict

from config import (
    LR_SCHEDULE_COSINE,
    LR_SCHEDULE_TRAPEZOIDAL,
    OPTIMIZER_ADAMW,
    OPTIMIZER_MUON,
)

if TYPE_CHECKING:
    from model import GPT


# ---------------------------------------------------------------------------
# Weight-decay mask
# ---------------------------------------------------------------------------


def decay_mask(params: Any) -> Any:
    """Weight-decay mask: apply decay only to tensors with ndim >= 2."""
    return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------


def _trapezoidal_schedule(
    cfg: ConfigDict, learning_rate: float | None = None,
) -> optax.Schedule:
    """Construct a trapezoidal (warmup → plateau → linear decay) LR schedule."""
    peak_lr = learning_rate if learning_rate is not None else cfg.learning_rate
    decay_start = cfg.max_steps - cfg.warmup_steps
    decay_steps = cfg.max_steps - decay_start  # decay fills the remaining steps
    warmup = optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=cfg.warmup_steps)
    plateau = optax.constant_schedule(value=peak_lr)
    decay = optax.linear_schedule(init_value=peak_lr, transition_steps=decay_steps, end_value=0.0)
    return optax.join_schedules(
        schedules=[warmup, plateau, decay],
        boundaries=[cfg.warmup_steps, decay_start],
    )


def build_lr_schedule(
    cfg: ConfigDict, learning_rate: float | None = None,
) -> optax.Schedule:
    """Create a learning rate schedule from the training configuration.

    Parameters
    ----------
    cfg:
        Training config with ``lr_schedule``, ``learning_rate``, ``warmup_steps``,
        ``max_steps``, and ``lr_end_ratio``.
    learning_rate:
        Override peak LR.  Defaults to ``cfg.learning_rate``.
    """
    peak_lr = learning_rate if learning_rate is not None else cfg.learning_rate

    if cfg.lr_schedule == LR_SCHEDULE_COSINE:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.max_steps - cfg.warmup_steps,
            end_value=peak_lr * cfg.lr_end_ratio,
        )

    if cfg.lr_schedule == LR_SCHEDULE_TRAPEZOIDAL:
        return _trapezoidal_schedule(cfg, learning_rate=peak_lr)

    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")


# ---------------------------------------------------------------------------
# AdamW builder
# ---------------------------------------------------------------------------


def build_adamw(
    schedule: optax.Schedule,
    *,
    b1: float = 0.90,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> optax.GradientTransformation:
    """Build an AdamW optimizer with 2D weight-decay masking."""
    return optax.adamw(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
        mask=decay_mask,
    )


# ---------------------------------------------------------------------------
# Muon parameter classification
# ---------------------------------------------------------------------------


def make_muon_weight_dimension_numbers(params: Any) -> Any:
    """Classify model parameters for Muon vs AdamW treatment.

    Matrix weights (MLP kernels, attention projection kernels) are assigned
    :class:`optax.contrib.MuonDimensionNumbers` for Muon orthogonalisation.
    Everything else (biases, embeddings, LM heads, norms) maps to ``None``
    and falls back to AdamW inside :func:`optax.contrib.muon`.

    Parameters
    ----------
    params:
        PyTree of model parameters.

    Returns
    -------
    PyTree matching ``params`` with ``MuonDimensionNumbers`` or ``None`` leaves.
    """
    def leaf_fn(path, val):
        # Scalars and 1D tensors (biases, norms) → Adam
        if getattr(val, "ndim", 0) < 2:
            return None

        # Extract path components as lowercase strings
        path_parts = []
        for p in path:
            if hasattr(p, "key"):
                path_parts.append(str(p.key).lower())
            elif hasattr(p, "name"):
                path_parts.append(str(p.name).lower())
            elif hasattr(p, "idx"):
                path_parts.append(str(p.idx).lower())
            else:
                path_parts.append(str(p).lower())
        if path_parts and path_parts[-1] == "value":
            path_parts = path_parts[:-1]

        # Skip biases
        if any("bias" in part for part in path_parts):
            return None

        # Skip embedding token projections
        if any(
            part in ("wte", "wpe", "embed", "embedding", "embeddings", "token_emb")
            or "embed" in part
            or "wte" in part
            or "wpe" in part
            for part in path_parts
        ):
            return None

        # Skip LM heads / output prediction heads
        if any(
            part in ("lm_head", "output_head", "head", "unembed", "logits")
            or "lm_head" in part
            or part.endswith("_head")
            for part in path_parts
        ):
            return None

        # 2D matrix weights (MLP layers, classical attention projections)
        if val.ndim == 2:
            return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)

        # 3D attention projection kernels (Flax / Flash attention)
        if val.ndim == 3:
            if any(part in ("out", "out_proj") for part in path_parts):
                return optax.contrib.MuonDimensionNumbers(reduction_axis=(0, 1), output_axis=2)
            if any(
                part in ("query", "key", "value", "q_proj", "k_proj", "v_proj")
                for part in path_parts
            ):
                return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=(1, 2))

        return None

    return jax.tree_util.tree_map_with_path(leaf_fn, params)


# ---------------------------------------------------------------------------
# Muon builder
# ---------------------------------------------------------------------------


def build_muon(
    schedule: optax.Schedule,
    *,
    cfg: ConfigDict | None = None,
    ns_steps: int = 5,
    beta: float = 0.95,
    weight_decay: float = 0.0,
    nesterov: bool = True,
    adam_b1: float = 0.90,
    adam_b2: float = 0.95,
    adam_weight_decay: float = 0.01,
    adam_learning_rate: float | optax.Schedule | None = None,
) -> optax.GradientTransformation:
    """Build a Muon optimizer (Keller Jordan, MomentUm Orthogonalized by Newton-Schulz).

    Applies Muon to matrix parameters and AdamW to everything else.

    Parameters
    ----------
    schedule:
        Learning rate schedule for the Muon branch.
    cfg:
        Training config (used to build the Adam LR schedule if
        ``adam_learning_rate`` is a float).
    ns_steps:
        Newton-Schulz iteration count.
    beta:
        Momentum coefficient for Muon.
    weight_decay:
        Weight decay for Muon-optimised parameters.
    nesterov:
        Whether to use Nesterov momentum.
    adam_b1, adam_b2, adam_weight_decay:
        AdamW hyperparameters for non-matrix parameters.
    adam_learning_rate:
        Learning rate (or schedule) for the AdamW branch.  If a float and
        ``cfg`` is provided, a matching schedule is built automatically.
    """
    muon_kwargs: dict[str, Any] = dict(
        learning_rate=schedule,
        ns_steps=ns_steps,
        beta=beta,
        weight_decay=weight_decay,
        nesterov=nesterov,
        muon_weight_dimension_numbers=make_muon_weight_dimension_numbers,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_weight_decay=adam_weight_decay,
    )

    # Resolve Adam learning rate
    if adam_learning_rate is None and cfg is not None:
        adam_learning_rate = getattr(
            getattr(cfg, "optimizer", ConfigDict()), "adam_learning_rate", 3e-4,
        )

    if adam_learning_rate is not None:
        if callable(adam_learning_rate):
            muon_kwargs["adam_learning_rate"] = adam_learning_rate
        elif isinstance(adam_learning_rate, (int, float)):
            if cfg is not None:
                muon_kwargs["adam_learning_rate"] = build_lr_schedule(cfg, learning_rate=float(adam_learning_rate))
            else:
                muon_kwargs["adam_learning_rate"] = float(adam_learning_rate)

    return optax.contrib.muon(**muon_kwargs)


# ---------------------------------------------------------------------------
# Parameter report
# ---------------------------------------------------------------------------


def print_optimizer_params(model: nnx.Module, cfg: ConfigDict) -> None:
    """Print each model parameter, its shape, and which optimizer updates it."""
    opt_cfg = getattr(cfg, "optimizer", ConfigDict())
    opt_type = str(opt_cfg.get("type", OPTIMIZER_ADAMW)).lower()

    params = nnx.state(model, nnx.Param)
    leaves_params = jax.tree_util.tree_leaves_with_path(params)

    if not leaves_params:
        print("No parameters found in model.")
        return

    print("\n" + "=" * 80)
    print(f"Optimizer Parameter Assignment (type={opt_type.upper()})")
    print("-" * 80)
    print(f"{'Parameter':<48} {'Shape':<18} {'Optimizer':<12}")
    print("-" * 80)

    muon_count = muon_elems = other_count = other_elems = 0

    if opt_type == OPTIMIZER_MUON:
        mask = make_muon_weight_dimension_numbers(params)
        is_leaf = lambda x: x is None or isinstance(x, optax.contrib.MuonDimensionNumbers)
        leaves_mask = jax.tree_util.tree_leaves_with_path(mask, is_leaf=is_leaf)
    else:
        leaves_mask = [(None, None)] * len(leaves_params)

    for (path, val), (_, m_val) in zip(leaves_params, leaves_mask):
        path_tokens = [
            str(getattr(x, "key", getattr(x, "name", getattr(x, "idx", x))))
            for x in path
        ]
        if path_tokens and path_tokens[-1] == "value":
            path_tokens = path_tokens[:-1]
        name = ".".join(path_tokens)
        shape = str(getattr(val, "shape", ()))

        assigned = "Muon" if (opt_type == OPTIMIZER_MUON and m_val is not None) else "AdamW"

        if assigned == "Muon":
            muon_count += 1
            muon_elems += getattr(val, "size", 0)
        else:
            other_count += 1
            other_elems += getattr(val, "size", 0)

        print(f"{name:<48} {shape:<18} {assigned:<12}")

    total = muon_elems + other_elems
    print("-" * 80)
    if opt_type == OPTIMIZER_MUON:
        print(
            f"Total: {len(leaves_params)} tensors ({total:,} params) | "
            f"Muon: {muon_count} ({muon_elems:,}) | AdamW: {other_count} ({other_elems:,})"
        )
    else:
        print(f"Total: {len(leaves_params)} tensors ({total:,} params) → all AdamW")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


def build_optimizer_tx(
    cfg: ConfigDict, schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Build an optax GradientTransformation from config.

    Parameters
    ----------
    cfg:
        Training configuration.
    schedule:
        Primary learning rate schedule.
    """
    opt_cfg = getattr(cfg, "optimizer", ConfigDict())
    opt_type = str(opt_cfg.get("type", OPTIMIZER_ADAMW)).lower()

    if opt_type == OPTIMIZER_MUON:
        tx = build_muon(
            schedule,
            cfg=cfg,
            beta=getattr(opt_cfg, "momentum", 0.95),
            weight_decay=getattr(opt_cfg, "muon_weight_decay", 0.0),
            adam_b1=getattr(opt_cfg, "b1", 0.90),
            adam_b2=getattr(opt_cfg, "b2", 0.95),
            adam_weight_decay=getattr(opt_cfg, "weight_decay", 0.01),
            adam_learning_rate=getattr(opt_cfg, "adam_learning_rate", 3e-4),
        )
    else:
        tx = build_adamw(
            schedule,
            b1=getattr(opt_cfg, "b1", 0.90),
            b2=getattr(opt_cfg, "b2", 0.95),
            eps=getattr(opt_cfg, "eps", 1e-8),
            weight_decay=getattr(opt_cfg, "weight_decay", 0.01),
        )

    # Optional gradient clipping
    clip_norm = getattr(opt_cfg, "clip_by_global_norm", None)
    if clip_norm is not None and clip_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(clip_norm), tx)

    # Gradient accumulation
    grad_acc_steps = getattr(cfg, "grad_acc_steps", 1)
    if grad_acc_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_acc_steps, use_grad_mean=True)

    return tx


def build_optimizer(
    model: GPT | nnx.Module,
    cfg: ConfigDict,
) -> tuple[nnx.Optimizer, optax.Schedule]:
    """Create a Flax NNX Optimizer and its learning rate schedule.

    Parameters
    ----------
    model:
        Flax NNX model.
    cfg:
        Training configuration.

    Returns
    -------
    (optimizer, schedule)
        The instantiated NNX optimizer and schedule function.
    """
    opt_type = str(getattr(cfg.optimizer, "type", OPTIMIZER_ADAMW)).lower()
    if opt_type == OPTIMIZER_MUON:
        # Muon uses cfg.learning_rate (0.02) for matrix params
        lr = cfg.learning_rate
    else:
        # Standalone AdamW uses the smaller adam_learning_rate
        lr = getattr(cfg.optimizer, "adam_learning_rate", 3e-4)
    schedule = build_lr_schedule(cfg, learning_rate=lr)
    tx = build_optimizer_tx(cfg, schedule)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, schedule
