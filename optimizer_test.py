"""
Tests for optimizer.py
======================

Verifies:
1. LR Schedule construction (cosine & trapezoidal).
2. Weight decay masking on parameter PyTrees.
3. Default AdamW optimizer building and parameter updates.
4. Custom optimizer registration via @register_optimizer and execution.
5. Gradient accumulation (MultiSteps) integration.
6. Gradient clipping integration.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from config import LRSchedule, OptimizerType, get_cpu_test_config
from model import GPT, train_step
from optimizer import (
    OPTIMIZER_REGISTRY,
    build_adamw,
    build_lr_schedule,
    build_optimizer,
    build_optimizer_tx,
    decay_mask,
    get_optimizer_builder,
    register_optimizer,
)


def test_decay_mask():
    params = {
        "kernel": jnp.ones((4, 4)),
        "bias": jnp.ones((4,)),
        "sub": {
            "scale": jnp.ones((4,)),
            "weight": jnp.ones((2, 3, 4)),
        },
    }
    mask = decay_mask(params)
    assert mask["kernel"] is True
    assert mask["bias"] is False
    assert mask["sub"]["scale"] is False
    assert mask["sub"]["weight"] is True


def test_build_lr_schedule_cosine():
    cfg = get_cpu_test_config()
    cfg.lr_schedule = LRSchedule.COSINE.value
    cfg.learning_rate = 1e-3
    cfg.warmup_steps = 10
    cfg.max_steps = 100
    cfg.lr_end_ratio = 0.1

    sched = build_lr_schedule(cfg)
    assert sched(0) == 0.0
    assert jnp.isclose(sched(10), 1e-3)
    assert jnp.isclose(sched(100), 1e-4)


def test_build_lr_schedule_trapezoidal():
    cfg = get_cpu_test_config()
    cfg.lr_schedule = LRSchedule.TRAPEZOIDAL.value
    cfg.learning_rate = 1e-3
    cfg.warmup_steps = 10
    cfg.max_steps = 100

    sched = build_lr_schedule(cfg)
    assert sched(0) == 0.0
    assert jnp.isclose(sched(10), 1e-3)
    assert jnp.isclose(sched(50), 1e-3)  # plateau
    assert jnp.isclose(sched(100), 0.0)


def test_adamw_registered_by_default():
    builder = get_optimizer_builder(OptimizerType.ADAMW)
    assert callable(builder)
    assert builder is build_adamw


def test_build_default_adamw_tx():
    cfg = get_cpu_test_config()
    sched = build_lr_schedule(cfg)
    tx = build_optimizer_tx(cfg, sched)
    assert hasattr(tx, "init") and hasattr(tx, "update")


def test_custom_optimizer_registration():
    # Register a new custom optimizer
    @register_optimizer("test_sgd")
    def build_test_sgd(schedule, momentum=0.9, **kwargs):
        return optax.sgd(learning_rate=schedule, momentum=momentum)

    assert "test_sgd" in OPTIMIZER_REGISTRY

    cfg = get_cpu_test_config()
    cfg.optimizer.type = "test_sgd"
    cfg.optimizer.momentum = 0.95

    sched = build_lr_schedule(cfg)
    tx = build_optimizer_tx(cfg, sched)
    assert hasattr(tx, "init") and hasattr(tx, "update")


def test_optimizer_unknown_raises():
    cfg = get_cpu_test_config()
    cfg.optimizer.type = "non_existent_optimizer"
    sched = build_lr_schedule(cfg)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer_tx(cfg, sched)


def test_gradient_accumulation_and_clipping():
    cfg = get_cpu_test_config()
    cfg.grad_acc_steps = 2
    cfg.optimizer.clip_by_global_norm = 1.0

    sched = build_lr_schedule(cfg)
    tx = build_optimizer_tx(cfg, sched)
    assert hasattr(tx, "init") and hasattr(tx, "update")


def test_full_train_step_with_adamw():
    cfg = get_cpu_test_config()
    cfg.model.vocab_size = 64

    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))
    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(0)
        model = GPT(cfg.model, rngs=rngs)
        optimizer, schedule = build_optimizer(model, cfg)

        B, T = 2, cfg.sequence_length
        x = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, cfg.model.vocab_size)
        y = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.model.vocab_size)

        # Initial loss
        loss_0 = train_step(model, optimizer, x, y)
        assert not jnp.isnan(loss_0)
        assert loss_0 > 0.0

        # Step again to ensure state updates smoothly
        loss_1 = train_step(model, optimizer, x, y)
        assert not jnp.isnan(loss_1)


def test_full_train_step_with_custom_optimizer():
    @register_optimizer("my_custom_adam")
    def build_my_custom_adam(schedule, **kwargs):
        # A custom variant using standard adam without weight decay
        return optax.adam(learning_rate=schedule, b1=0.88, b2=0.98)

    cfg = get_cpu_test_config()
    cfg.model.vocab_size = 64
    cfg.optimizer.type = "my_custom_adam"

    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))
    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(1)
        model = GPT(cfg.model, rngs=rngs)

        optimizer, schedule = build_optimizer(model, cfg)

        B, T = 2, cfg.sequence_length
        x = jax.random.randint(jax.random.PRNGKey(2), (B, T), 0, cfg.model.vocab_size)
        y = jax.random.randint(jax.random.PRNGKey(3), (B, T), 0, cfg.model.vocab_size)

        loss_0 = train_step(model, optimizer, x, y)
        assert not jnp.isnan(loss_0)
        assert loss_0 > 0.0
