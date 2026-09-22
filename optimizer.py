"""
Optimizer utilities and custom optimizer support for nano-GPT JAX.

Provides:
- An extensible optimizer registry (:func:`register_optimizer`, :data:`OPTIMIZER_REGISTRY`).
- The default AdamW optimizer builder with 2D weight decay masking (:func:`build_adamw`).
- Learning rate schedule builder (:func:`build_lr_schedule`).
- High-level optimizer builders (:func:`build_optimizer_tx`, :func:`build_optimizer`).

Custom optimizers can be registered using the ``@register_optimizer`` decorator:

.. code-block:: python

    from optimizer import register_optimizer
    import optax

    @register_optimizer("sgd")
    def build_sgd(schedule, momentum=0.9, **kwargs):
        return optax.sgd(learning_rate=schedule, momentum=momentum)

Then selected via config or CLI:

    python train.py --set optimizer.type=sgd --set optimizer.momentum=0.95
"""

from __future__ import annotations

import inspect
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

import jax
import optax
from flax import nnx
from ml_collections import ConfigDict

from config import LRSchedule, OptimizerType

if TYPE_CHECKING:
    from model import GPT


# Type alias for optimizer builder functions.
# A builder accepts (schedule, **kwargs) or (schedule, cfg, **kwargs)
# and returns an optax.GradientTransformation.
OptimizerBuilder = Callable[..., optax.GradientTransformation]

OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {}


def register_optimizer(name: str | OptimizerType) -> Callable[[OptimizerBuilder], OptimizerBuilder]:
    """Decorator to register a custom optimizer builder.

    Parameters
    ----------
    name:
        The string name or OptimizerType to associate with the builder.
    """
    key = str(name.value if isinstance(name, OptimizerType) else name).lower()

    def decorator(fn: OptimizerBuilder) -> OptimizerBuilder:
        OPTIMIZER_REGISTRY[key] = fn
        return fn

    return decorator


def get_optimizer_builder(name: str | OptimizerType) -> OptimizerBuilder:
    """Retrieve an optimizer builder function from the registry."""
    key = str(name.value if isinstance(name, OptimizerType) else name).lower()
    if key not in OPTIMIZER_REGISTRY:
        valid = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer '{name}'. Registered optimizers: {valid}")
    return OPTIMIZER_REGISTRY[key]


def decay_mask(params: Any) -> Any:
    """Decay mask that applies weight decay only to tensors with ndim >= 2 (e.g. weights, not biases or scales)."""
    return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)


def trapezoidal_schedule(config: ConfigDict, learning_rate: float | None = None) -> optax.Schedule:
    """Construct a trapezoidal learning rate schedule."""
    peak_lr = learning_rate if learning_rate is not None else config.learning_rate
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=config.warmup_steps,
    )
    plateau_schedule = optax.constant_schedule(value=peak_lr)
    decay_schedule = optax.linear_schedule(
        init_value=peak_lr,
        transition_steps=config.warmup_steps,
        end_value=0.0,
    )

    return optax.join_schedules(
        schedules=[warmup_schedule, plateau_schedule, decay_schedule],
        boundaries=[config.warmup_steps, config.max_steps - config.warmup_steps],
    )


def build_lr_schedule(cfg: ConfigDict, learning_rate: float | None = None) -> optax.Schedule:
    """Create a learning rate schedule from the provided training configuration."""
    schedule_type = cfg.lr_schedule
    peak_lr = learning_rate if learning_rate is not None else cfg.learning_rate
    if schedule_type == LRSchedule.COSINE:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.max_steps - cfg.warmup_steps,
            end_value=peak_lr * cfg.lr_end_ratio,
        )
    elif schedule_type == LRSchedule.TRAPEZOIDAL:
        return trapezoidal_schedule(cfg, learning_rate=peak_lr)
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")


@register_optimizer(OptimizerType.ADAMW)
def build_adamw(
    schedule: optax.Schedule,
    *,
    b1: float = 0.90,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    mask: Any | None = decay_mask,
    **kwargs,
) -> optax.GradientTransformation:
    """Default AdamW optimizer matching the original nano-GPT JAX configuration."""
    return optax.adamw(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        weight_decay=weight_decay,
        mask=mask,
    )


def make_muon_weight_dimension_numbers(params: Any) -> Any:
    """Dimension numbers mask that applies Muon only to matrices (excluding biases, heads, and embeddings).

    Parameters for which this function returns None fall back to Adam in :func:`optax.contrib.muon`.

    Parameters
    ----------
    params:
        The PyTree of model parameters.

    Returns
    -------
    A PyTree with the same structure as ``params``, where internal weight
    matrices map to :class:`optax.contrib.MuonDimensionNumbers` and biases,
    embeddings, output heads, and non-matrix tensors map to ``None``.
    """
    def leaf_fn(path, val):
        # Biases, layernorms, scales, scalars (ndim < 2) -> Adam
        if getattr(val, "ndim", 0) < 2:
            return None

        # Extract path components as lowercase strings (strip trailing NNX '.value')
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

        # 1. Skip biases (e.g. 1D biases or 2D attention biases like (num_heads, head_dim))
        if any("bias" in part for part in path_parts):
            return None

        # 2. Skip embedding token projections (e.g. wte, wpe, embed, embedding)
        if any(
            part in ("wte", "wpe", "embed", "embedding", "embeddings", "token_emb")
            or "embed" in part
            or "wte" in part
            or "wpe" in part
            for part in path_parts
        ):
            return None

        # 3. Skip LM heads / output prediction heads (e.g. lm_head, output_head, head)
        if any(
            part in ("lm_head", "output_head", "head", "unembed", "logits")
            or "lm_head" in part
            or part.endswith("_head")
            for part in path_parts
        ):
            return None

        # 4. Standard 2D matrix weights (e.g. MLP layers, classical attention projections)
        if val.ndim == 2:
            return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)

        # 5. 3D attention projection kernels (Flax / Flash attention)
        if val.ndim == 3:
            # Output projection: shape is (num_heads, head_dim, n_embd) -> reduce (0, 1) to out 2
            if any(part in ("out", "out_proj") for part in path_parts):
                return optax.contrib.MuonDimensionNumbers(reduction_axis=(0, 1), output_axis=2)
            # Query, Key, Value projections: shape is (n_embd, num_heads, head_dim) -> reduce 0 to out (1, 2)
            if any(
                part in ("query", "key", "value", "q_proj", "k_proj", "v_proj")
                for part in path_parts
            ):
                return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=(1, 2))

        return None

    return jax.tree_util.tree_map_with_path(leaf_fn, params)




muon_weight_dimension_numbers = make_muon_weight_dimension_numbers


@register_optimizer(OptimizerType.MUON)
def build_muon(
    schedule: optax.Schedule,
    *,
    cfg: ConfigDict | None = None,
    ns_steps: int = 5,
    beta: float = 0.95,
    weight_decay: float = 0.0,
    nesterov: bool = True,
    muon_weight_dimension_numbers: Any | None = make_muon_weight_dimension_numbers,
    **kwargs,
) -> optax.GradientTransformation:
    """Muon optimizer builder using optax.contrib.muon.

    Implements Keller Jordan's Muon optimizer (MomentUm Orthogonalized by Newton-schulz).
    By default, applies Muon only to matrix parameters (excluding biases, embedding token
    projections, and output heads, which are optimized with AdamW).
    """
    if "momentum" in kwargs:
        beta = kwargs["momentum"]
    elif "beta" in kwargs:
        beta = kwargs["beta"]

    if "muon_weight_decay" in kwargs:
        weight_decay = kwargs["muon_weight_decay"]

    if muon_weight_dimension_numbers is None:
        muon_weight_dimension_numbers = make_muon_weight_dimension_numbers

    # Forward any supported extra hyperparameters to optax.contrib.muon
    muon_kwargs = {
        "learning_rate": schedule,
        "ns_steps": ns_steps,
        "beta": beta,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
    }

    # AdamW hyperparameters for non-matrix parameters (embeddings, heads, biases, RMSNorm scales)
    adam_b1 = kwargs.get("adam_b1", kwargs.get("b1", 0.90))
    adam_b2 = kwargs.get("adam_b2", kwargs.get("b2", 0.95))
    adam_wd = kwargs.get("adam_weight_decay", kwargs.get("weight_decay", 0.01))
    muon_kwargs["adam_b1"] = adam_b1
    muon_kwargs["adam_b2"] = adam_b2
    muon_kwargs["adam_weight_decay"] = adam_wd

    # Auxiliary AdamW learning rate (defaults to 3e-4, scheduled across training steps if cfg available)
    adam_lr = kwargs.get("adam_learning_rate", None)
    if adam_lr is None and cfg is not None:
        opt_cfg = getattr(cfg, "optimizer", ConfigDict())
        adam_lr = getattr(opt_cfg, "adam_learning_rate", 3e-4)

    if adam_lr is not None:
        if callable(adam_lr):
            muon_kwargs["adam_learning_rate"] = adam_lr
        elif isinstance(adam_lr, (int, float)):
            if cfg is not None:
                muon_kwargs["adam_learning_rate"] = build_lr_schedule(cfg, learning_rate=float(adam_lr))
            else:
                muon_kwargs["adam_learning_rate"] = float(adam_lr)

    for k in (
        "adam_eps_root",
        "eps",
        "mu_dtype",
        "preconditioning",
        "consistent_rms",
    ):
        if k in kwargs:
            muon_kwargs[k] = kwargs[k]

    return optax.contrib.muon(**muon_kwargs)


def print_optimizer_params(model: nnx.Module, cfg: ConfigDict) -> None:
    """Print out each model parameter, its shape, and which optimizer updates it.

    Parameters
    ----------
    model:
        The Flax NNX model containing parameters.
    cfg:
        Configuration dict with optimizer configuration.
    """
    opt_cfg = getattr(cfg, "optimizer", ConfigDict())
    opt_type = str(opt_cfg.get("type", OptimizerType.ADAMW.value)).lower()

    params = nnx.state(model, nnx.Param)
    leaves_params = jax.tree_util.tree_leaves_with_path(params)

    if not leaves_params:
        print("No parameters found in model.")
        return

    print("\n" + "=" * 80)
    print(f"Optimizer Parameter Assignment (Configured: {opt_type.upper()})")
    print("-" * 80)
    if opt_type == "muon":
        muon_lr = getattr(opt_cfg, "muon_learning_rate", getattr(cfg, "learning_rate", 0.02))
        momentum = getattr(opt_cfg, "momentum", getattr(opt_cfg, "beta", 0.95))
        muon_wd = getattr(opt_cfg, "muon_weight_decay", 0.0)
        adam_lr = getattr(opt_cfg, "adam_learning_rate", 3e-4)
        adam_b1 = getattr(opt_cfg, "adam_b1", getattr(opt_cfg, "b1", 0.90))
        adam_b2 = getattr(opt_cfg, "adam_b2", getattr(opt_cfg, "b2", 0.95))
        adam_wd = getattr(opt_cfg, "adam_weight_decay", getattr(opt_cfg, "weight_decay", 0.01))
        print(f"Muon:  lr={muon_lr}, momentum={momentum}, weight_decay={muon_wd}")
        print(f"AdamW: lr={adam_lr}, betas=({adam_b1}, {adam_b2}), weight_decay={adam_wd}")
        print("-" * 80)
    elif opt_type == "adamw":
        adam_lr = getattr(opt_cfg, "adam_learning_rate", getattr(cfg, "learning_rate", 3e-4))
        b1 = getattr(opt_cfg, "b1", 0.90)
        b2 = getattr(opt_cfg, "b2", 0.95)
        wd = getattr(opt_cfg, "weight_decay", 0.01)
        print(f"AdamW: lr={adam_lr}, betas=({b1}, {b2}), weight_decay={wd}")
        print("-" * 80)
    print(f"{'Parameter':<48} {'Shape':<18} {'Optimizer':<12}")
    print("-" * 80)

    muon_counts = 0
    muon_elements = 0
    other_counts = 0
    other_elements = 0
    other_opt_name = "AdamW" if opt_type == "muon" else opt_type.capitalize()

    if opt_type == "muon":
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
        param_name = ".".join(path_tokens)
        shape_str = str(getattr(val, "shape", ()))

        if opt_type == "muon":
            assigned_opt = "Muon" if m_val is not None else "AdamW"
        else:
            assigned_opt = other_opt_name

        if assigned_opt == "Muon":
            muon_counts += 1
            muon_elements += getattr(val, "size", 0)
        else:
            other_counts += 1
            other_elements += getattr(val, "size", 0)

        print(f"{param_name:<48} {shape_str:<18} {assigned_opt:<12}")

    total_params = muon_elements + other_elements
    print("-" * 80)
    if opt_type == "muon":
        print(
            f"Total: {len(leaves_params)} tensors ({total_params:,} params) | "
            f"Muon: {muon_counts} tensors ({muon_elements:,} params) | "
            f"AdamW: {other_counts} tensors ({other_elements:,} params)"
        )
    else:
        print(f"Total: {len(leaves_params)} tensors ({total_params:,} params) -> all using {other_opt_name}")
    print("=" * 80 + "\n")


def build_optimizer_tx(
    cfg: ConfigDict,
    schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Build an :class:`optax.GradientTransformation` according to configuration.

    Parameters
    ----------
    cfg:
        Training configuration dict.
    schedule:
        Learning rate schedule.
    """
    opt_cfg = getattr(cfg, "optimizer", ConfigDict())
    opt_type = opt_cfg.get("type", OptimizerType.ADAMW.value)

    builder = get_optimizer_builder(opt_type)

    # Convert ConfigDict to regular dict of kwargs, omitting 'type' and internal metadata
    opt_kwargs = dict(opt_cfg.to_dict()) if hasattr(opt_cfg, "to_dict") else dict(opt_cfg)
    opt_kwargs.pop("type", None)

    clip_norm = opt_kwargs.pop("clip_by_global_norm", None)

    # Check builder signature: pass cfg if it accepts 'cfg' parameter
    sig = inspect.signature(builder)
    if "cfg" in sig.parameters:
        opt_kwargs["cfg"] = cfg

    # If the function does not accept **kwargs, filter down to accepted parameters
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_keyword:
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in sig.parameters}

    tx = builder(schedule, **opt_kwargs)

    # Optional gradient clipping before optimizer updates
    chains = []
    if clip_norm is not None and clip_norm > 0:
        chains.append(optax.clip_by_global_norm(clip_norm))
    chains.append(tx)

    if len(chains) > 1:
        tx = optax.chain(*chains)

    # Gradient accumulation if configured
    grad_acc_steps = getattr(cfg, "grad_acc_steps", 1)
    if grad_acc_steps > 1:
        tx = optax.MultiSteps(
            tx, every_k_schedule=grad_acc_steps, use_grad_mean=True
        )

    return tx


def build_optimizer(
    model: GPT | nnx.Module,
    cfg: ConfigDict,
    schedule: optax.Schedule | None = None,
) -> tuple[nnx.Optimizer, optax.Schedule]:
    """Create a Flax NNX Optimizer and its corresponding schedule.

    Parameters
    ----------
    model:
        Flax NNX model (e.g. :class:`~model.GPT`).
    cfg:
        Configuration dictionary containing optimizer and LR schedule options.
    schedule:
        Optional pre-built schedule. If None, built via :func:`build_lr_schedule(cfg)`.

    Returns
    -------
    tuple[nnx.Optimizer, optax.Schedule]
        The instantiated NNX optimizer and schedule function.
    """
    if schedule is None:
        opt_cfg = getattr(cfg, "optimizer", ConfigDict())
        opt_type = opt_cfg.get("type", OptimizerType.ADAMW.value)
        if opt_type == OptimizerType.ADAMW.value and getattr(cfg, "learning_rate", None) == 0.02:
            adam_lr = getattr(opt_cfg, "adam_learning_rate", 3e-4)
            schedule = build_lr_schedule(cfg, learning_rate=adam_lr)
        else:
            schedule = build_lr_schedule(cfg)

    tx = build_optimizer_tx(cfg, schedule)

    optimizer = nnx.Optimizer(
        model,
        tx,
        wrt=nnx.Param,
    )
    return optimizer, schedule
