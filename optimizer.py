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


def trapezoidal_schedule(config: ConfigDict) -> optax.Schedule:
    """Construct a trapezoidal learning rate schedule."""
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    plateau_schedule = optax.constant_schedule(value=config.learning_rate)
    decay_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        transition_steps=config.warmup_steps,
        end_value=0.0,
    )

    return optax.join_schedules(
        schedules=[warmup_schedule, plateau_schedule, decay_schedule],
        boundaries=[config.warmup_steps, config.max_steps - config.warmup_steps],
    )


def build_lr_schedule(cfg: ConfigDict) -> optax.Schedule:
    """Create a learning rate schedule from the provided training configuration."""
    schedule_type = cfg.lr_schedule
    if schedule_type == LRSchedule.COSINE:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.max_steps - cfg.warmup_steps,
            end_value=cfg.learning_rate * cfg.lr_end_ratio,
        )
    elif schedule_type == LRSchedule.TRAPEZOIDAL:
        return trapezoidal_schedule(cfg)
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")


@register_optimizer(OptimizerType.ADAMW)
def build_adamw(
    schedule: optax.Schedule,
    *,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
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
        schedule = build_lr_schedule(cfg)

    tx = build_optimizer_tx(cfg, schedule)

    optimizer = nnx.Optimizer(
        model,
        tx,
        wrt=nnx.Param,
    )
    return optimizer, schedule
