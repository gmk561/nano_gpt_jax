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
    build_muon,
    build_optimizer,
    build_optimizer_tx,
    decay_mask,
    get_optimizer_builder,
    make_muon_weight_dimension_numbers,
    muon_weight_dimension_numbers,
    print_optimizer_params,
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


def test_muon_registered_by_default():
    builder = get_optimizer_builder(OptimizerType.MUON)
    assert callable(builder)
    assert builder is build_muon


def test_muon_step_dummy_input():
    # 1. Verify build_optimizer_tx with OptimizerType.MUON
    cfg = get_cpu_test_config()
    cfg.optimizer.type = OptimizerType.MUON.value
    schedule = optax.constant_schedule(1e-3)
    tx = build_optimizer_tx(cfg, schedule)
    assert hasattr(tx, "init") and hasattr(tx, "update")

    # 2. Setup dummy parameters (2D matrix for Muon orthogonalization, 1D vector for Adam)
    params = {
        "layer": {
            "weight": jnp.ones((16, 8)),
            "bias": jnp.zeros((8,)),
        }
    }
    grads = {
        "layer": {
            "weight": jnp.ones((16, 8)) * 0.1,
            "bias": jnp.ones((8,)) * 0.1,
        }
    }

    state = tx.init(params)

    # 3. Test compilation (jax.jit) and applying one step of dummy input
    @jax.jit
    def step_fn(p, s, g):
        updates, new_s = tx.update(g, s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_s

    new_params, new_state = step_fn(params, state, grads)

    # 4. Sanity checks on shapes, validity, and actual parameter updates
    assert new_params["layer"]["weight"].shape == params["layer"]["weight"].shape
    assert new_params["layer"]["bias"].shape == params["layer"]["bias"].shape
    assert not jnp.isnan(new_params["layer"]["weight"]).any()
    assert not jnp.isnan(new_params["layer"]["bias"]).any()
    assert not jnp.isinf(new_params["layer"]["weight"]).any()
    assert not jnp.isinf(new_params["layer"]["bias"]).any()
    assert not jnp.allclose(new_params["layer"]["weight"], params["layer"]["weight"])
    assert not jnp.allclose(new_params["layer"]["bias"], params["layer"]["bias"])


def test_muon_weight_dimension_numbers_mask():
    params = {
        "blocks": {
            "0": {
                "mha": {
                    "query": {
                        "kernel": jnp.ones((16, 2, 8)),
                        "bias": jnp.zeros((2, 8)),  # 2D bias in multi-head attention
                    },
                    "out": {
                        "kernel": jnp.ones((2, 8, 16)),
                        "bias": jnp.zeros((16,)),
                    },
                },
                "mlp": {"kernel": jnp.ones((16, 64)), "bias": jnp.zeros((64,))},
            }
        },
        "wte": {"embedding": jnp.ones((100, 16))},
        "wpe": {"embedding": jnp.ones((32, 16))},
        "lm_head": {"kernel": jnp.ones((16, 100))},
        "ln_f": {"scale": jnp.ones((16,))},
    }

    mask = muon_weight_dimension_numbers(params)

    # Matrix weights should have MuonDimensionNumbers
    assert isinstance(mask["blocks"]["0"]["mha"]["query"]["kernel"], optax.contrib.MuonDimensionNumbers)
    assert isinstance(mask["blocks"]["0"]["mha"]["out"]["kernel"], optax.contrib.MuonDimensionNumbers)
    assert isinstance(mask["blocks"]["0"]["mlp"]["kernel"], optax.contrib.MuonDimensionNumbers)

    # Biases (both 1D and 2D) and 1D scales should be None (optimized with AdamW)
    assert mask["blocks"]["0"]["mha"]["query"]["bias"] is None
    assert mask["blocks"]["0"]["mha"]["out"]["bias"] is None
    assert mask["blocks"]["0"]["mlp"]["bias"] is None
    assert mask["ln_f"]["scale"] is None

    # Embeddings and output heads should be None (optimized with AdamW)
    assert mask["wte"]["embedding"] is None
    assert mask["wpe"]["embedding"] is None
    assert mask["lm_head"]["kernel"] is None



def test_muon_dimension_numbers_selective_updates():
    schedule = optax.constant_schedule(1e-3)
    # build_muon uses muon_weight_dimension_numbers by default
    tx = build_muon(schedule)

    params = {
        "blocks": {
            "0": {
                "mlp": {"w": jnp.ones((8, 16)), "b": jnp.zeros((16,))},
            }
        },
        "wte": {"embedding": jnp.ones((64, 8))},
        "lm_head": {"kernel": jnp.ones((8, 64))},
    }
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.05, params)

    state = tx.init(params)

    @jax.jit
    def step_fn(p, s, g):
        updates, new_s = tx.update(g, s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_s

    new_params, new_state = step_fn(params, state, grads)

    # Ensure all parameter groups are updated without NaNs
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_params):
        assert not jnp.isnan(leaf).any()

    # Matrix weights in blocks receive Muon updates
    assert not jnp.allclose(new_params["blocks"]["0"]["mlp"]["w"], params["blocks"]["0"]["mlp"]["w"])
    # Biases receive Adam updates
    assert not jnp.allclose(new_params["blocks"]["0"]["mlp"]["b"], params["blocks"]["0"]["mlp"]["b"])
    # Embeddings receive Adam updates
    assert not jnp.allclose(new_params["wte"]["embedding"], params["wte"]["embedding"])
    # Heads receive Adam updates
    assert not jnp.allclose(new_params["lm_head"]["kernel"], params["lm_head"]["kernel"])


def test_print_optimizer_params(capsys):
    class DummyBlock(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(8, 16, rngs=rngs)

    class DummyModel(nnx.Module):
        def __init__(self, rngs):
            self.wte = nnx.Embed(32, 8, rngs=rngs)
            self.block = DummyBlock(rngs)
            self.lm_head = nnx.Linear(8, 32, use_bias=False, rngs=rngs)

    model = DummyModel(nnx.Rngs(0))

    cfg = get_cpu_test_config()
    cfg.optimizer.type = OptimizerType.MUON.value

    print_optimizer_params(model, cfg)
    captured = capsys.readouterr().out
    assert "Optimizer Parameter Assignment (Configured: MUON)" in captured
    assert "Muon" in captured
    assert "AdamW" in captured

    cfg.optimizer.type = OptimizerType.ADAMW.value
    print_optimizer_params(model, cfg)
    captured_adam = capsys.readouterr().out
    assert "Optimizer Parameter Assignment (Configured: ADAMW)" in captured_adam




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


def test_full_train_step_with_muon():
    cfg = get_cpu_test_config()
    cfg.model.vocab_size = 64
    cfg.optimizer.type = OptimizerType.MUON.value

    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))
    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(42)
        model = GPT(cfg.model, rngs=rngs)

        optimizer, schedule = build_optimizer(model, cfg)

        B, T = 2, cfg.sequence_length
        x = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, cfg.model.vocab_size)
        y = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.model.vocab_size)

        loss_0 = train_step(model, optimizer, x, y)
        assert not jnp.isnan(loss_0)
        assert loss_0 > 0.0

        loss_1 = train_step(model, optimizer, x, y)
        assert not jnp.isnan(loss_1)

