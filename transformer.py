import os

# Load environment variables from local .env if it exists before JAX is imported
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Emulate 8 CPU devices by default if no XLA_FLAGS are set
# if "XLA_FLAGS" not in os.environ:
#     print("Setting XLA_FLAGS for 8 CPU devices")
#     os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import optax
import orbax.checkpoint as ocp
from flax import nnx
from ml_collections import ConfigDict
import functools

from data import (
    GPT2_VOCAB_SIZE,
    create_train_loader,
    create_val_loader,
    get_iter_state,
    restore_iter_state,
)

print("jax.device_count():", jax.device_count())

nnx.use_eager_sharding(True)


class MockWandb:
    @staticmethod
    def init(*args, **kwargs):
        class Run:
            def log(self, *args, **kwargs):
                pass

            def finish(self, *args, **kwargs):
                pass

        return Run()

    @staticmethod
    def log(*args, **kwargs):
        pass

    @staticmethod
    def finish(*args, **kwargs):
        pass


if os.environ.get("WANDB_MODE") == "disabled":
    wandb = MockWandb()
else:
    import wandb


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.sequence_length = 1024
    # Mixed precision: use bfloat16 for faster computation, weights remain float32, activations are bfloat16
    # don't apply to LayerNorm as it sums many values, this can lead to overflow or underflow.
    cfg.apply_dtype_policy = True
    cfg.num_devices = jax.device_count()
    cfg.device_batch_size = 32
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.gpt_batch_size = 524288
    cfg.grad_acc_steps = cfg.gpt_batch_size // (cfg.batch_size * cfg.sequence_length)
    cfg.dataset = "edu_fineweb"  # "input_txt" or "edu_fineweb"
    cfg.val_check_steps = 100  # evaluate validation loss every 100 steps
    cfg.val_max_steps = 50  # max number of batches to use for validation
    cfg.max_steps = 10_000  # total training steps
    cfg.warmup_steps = 715

    # Checkpointing config
    cfg.ckpt_dir_name = "checkpoints"
    cfg.ckpt_dir = os.path.join(os.path.dirname(__file__), cfg.ckpt_dir_name)
    cfg.ckpt_every_steps = (
        cfg.val_check_steps * 1
    )  # save every N effective training steps, best align with validation checks to have the validation loss computed for the checkpoint.
    cfg.ckpt_max_to_keep = 3  # keep N most recent + best val_loss
    cfg.resume_ckpt = None  # step number, "latest", or None to start fresh

    # GPT model config
    cfg.model = ConfigDict()
    cfg.model.block_size = cfg.sequence_length
    cfg.model.vocab_size = -1
    cfg.model.n_layer = 6
    cfg.model.n_head = 6
    cfg.model.n_embd = 384
    cfg.model.param_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.compute_dtype = jnp.bfloat16 if cfg.apply_dtype_policy else jnp.float32
    cfg.model.accum_dtype = jnp.float32
    return cfg


def get_cpu_test_config() -> ConfigDict:
    """Lightweight config for quick testing on CPU."""
    cfg = get_config()
    cfg.apply_dtype_policy = False
    cfg.device_batch_size = 1
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.gpt_batch_size = cfg.batch_size
    cfg.grad_acc_steps = 1
    cfg.val_check_steps = 10  # evaluate validation loss every 10 steps
    cfg.val_max_steps = 4  # max number of batches to use for validation
    cfg.max_steps = 200  # total training steps
    cfg.warmup_steps = 10
    # cfg.resume_ckpt = "latest"  # step number, "latest", or None to start fresh
    cfg.resume_ckpt = 100  # step number, "latest", or None to start fresh

    cfg.model.n_layer = 3
    cfg.model.n_head = 2
    cfg.model.n_embd = 16

    cfg.dataset = "input_txt"

    return cfg


_accelerator_backends = {"gpu", "tpu"}


class MLP(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
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


def is_cudnn_available():
    try:
        from jax._src.lib import cuda_versions

        return (
            cuda_versions is not None and cuda_versions.cudnn_get_version() is not None
        )
    except (ImportError, AttributeError, RuntimeError):
        return False


def flash_attention_fn(
    query, key, value, bias=None, mask=None, is_causal=True, **kwargs
):
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


class Block(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config
        init_fn = nnx.initializers.normal(stddev=0.02)
        # Sharding: all (None,...) = replicated for data parallelism.
        # For model parallelism later: QKV kernel -> (None, None, 'model'),
        #   out kernel -> (None, 'model', None).
        self.mha = nnx.MultiHeadAttention(
            num_heads=config.n_head,
            in_features=config.n_embd,
            qkv_features=config.n_embd,  # total dim; Flax splits by num_heads internally
            rngs=rngs,
            decode=False,
            dtype=config.compute_dtype,
            kernel_init=init_fn,
            kernel_metadata={"out_sharding": (None, None, None)},
            out_kernel_init=init_fn,
            out_kernel_metadata={"out_sharding": (None, None, None)},
            bias_init=nnx.initializers.zeros_init(),
            bias_metadata={"out_sharding": (None,)},
            out_bias_init=nnx.initializers.zeros_init(),
            out_bias_metadata={"out_sharding": (None,)},
            attention_fn=functools.partial(flash_attention_fn, is_causal=True),
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
        self.wpe = nnx.Embed(
            config.block_size,
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
        # Pre-compute causal mask once for the full block_size; slice at call time.
        self._causal_mask = jnp.tril(
            jnp.ones((1, 1, config.block_size, config.block_size), dtype=jnp.bool_)
        )
        self._init_weights(rngs)

    def _init_weights(self, rngs: nnx.Rngs):
        residual_scale = 1.0 / (2 * self.config.n_layer) ** 0.5

        def _is_residual_output(module, parent, attr_name):
            """Check if this Linear is the output projection of a residual branch."""
            # MLP's second linear (projects back into residual stream)
            if isinstance(parent, MLP) and attr_name == "linear_2":
                return True
            # MultiHeadAttention's output projection
            if isinstance(parent, nnx.MultiHeadAttention) and attr_name == "out":
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
        pos = jnp.arange(T)
        x = self.wte(x, out_sharding=jax.typeof(x).sharding) + self.wpe(
            pos
        )  # (B, T, n_embd)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x).astype(self.config.compute_dtype)
        # weight tying: reuse wte embedding matrix as output projection
        logits = x @ self.wte.embedding[...].T  # (B, T, vocab_size)

        return logits.astype(self.config.accum_dtype)


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> char

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[i] for i in tokens)


# DataLoader and EduFinewebDataLoader have been moved to data.py (Grain-based).
# Use create_train_loader() / create_val_loader() from that module instead.


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
    for path, module in nnx.iter_modules(model):
        for attr in ("kernel", "embedding", "scale", "bias"):
            param = getattr(module, attr, None)
            if param is not None and hasattr(param, "value"):
                dtype = param[...].dtype
                if dtype == jnp.float32:
                    print(f"{path} {attr} {dtype}")


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
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)

    optimizer.update(model, grads)
    return loss


@nnx.jit
def val_step(model: GPT, x: jnp.ndarray, y: jnp.ndarray):
    loss = loss_fn(model, x, y)
    return loss


def align_acc_step(step: int, gradient_acc_steps: int) -> int:
    return step // gradient_acc_steps


# ── Checkpointing helpers ────────────────────────────────────────────────────


def get_checkpoint_state(
    model: GPT,
    optimizer: nnx.Optimizer,
    step: int,
) -> dict:
    """Extract a checkpoint-friendly PyTree from NNX modules.

    Data-loader position is **not** stored here; it is written separately as a
    Grain state file (``grain_train_state.bin``) alongside each Orbax
    checkpoint directory.  Use :func:`get_iter_state` /
    :func:`restore_iter_state` from ``data.py`` for that.
    """
    return {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": step,
    }


def restore_checkpoint_state(
    model: GPT,
    optimizer: nnx.Optimizer,
    state_dict,
) -> int:
    """Restore NNX module state from a loaded checkpoint PyTree.

    Returns
    -------
    int
        The effective training step stored in the checkpoint.
    """
    nnx.update(model, state_dict["model"])
    nnx.update(optimizer, state_dict["optimizer"])
    return int(state_dict["step"])


if __name__ == "__main__":
    # ── Training setup ──────────────────────────────────────────────────────────
    _has_accelerator = any(d.platform in _accelerator_backends for d in jax.devices())
    print(f"Training with accelerator: {_has_accelerator}")
    print("Is cudnn available? ", is_cudnn_available())

    cfg = get_config() if _has_accelerator else get_cpu_test_config()
    # 2D mesh: ('data', 'model'). For now model=1; later change to (dp, mp) for tensor parallelism.
    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))

    # Build Grain data loaders (support multi-host data parallelism via
    # ShardByJaxProcess; on a single process this is equivalent to NoSharding).
    train_loader = create_train_loader(cfg)
    val_loader = create_val_loader(cfg)  # None for input_txt dataset

    # Both datasets use the GPT-2 tokenizer.
    cfg.model.vocab_size = GPT2_VOCAB_SIZE

    # Create a persistent training iterator so we can checkpoint its position.
    train_iter = iter(train_loader)

    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(0)
        model = GPT(cfg.model, rngs=rngs)

        if cfg.apply_dtype_policy:
            apply_dtype_policy(model, cfg.model)

        dtype_report(model)

        learning_rate = 6e-4
        decay_steps = cfg.max_steps - cfg.warmup_steps

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=decay_steps,
            end_value=learning_rate * 0.1,
        )

        def decay_mask(params):
            return jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                schedule,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=0.1,
                mask=decay_mask,
            ),
        )

        if cfg.grad_acc_steps > 1:
            tx = optax.MultiSteps(
                tx, every_k_schedule=cfg.grad_acc_steps, use_grad_mean=True
            )

        optimizer = nnx.Optimizer(
            model,
            tx,
            wrt=nnx.Param,
        )

        # ── Checkpointing ────────────────────────────────────────────────────────────

        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg.ckpt_max_to_keep,
            best_fn=lambda metrics: metrics["val_loss"],
            best_mode="min",
            enable_async_checkpointing=True,
        )
        ckpt_mngr = ocp.CheckpointManager(
            cfg.ckpt_dir,
            options=ckpt_options,
        )

        # Resume from checkpoint if configured
        start_step = 0
        resume = cfg.resume_ckpt
        if resume is not None:
            restore_step = (
                ckpt_mngr.latest_step() if resume == "latest" else int(resume)
            )
            if restore_step is not None:
                abstract_state = get_checkpoint_state(model, optimizer, 0)
                restored = ckpt_mngr.restore(
                    restore_step,
                    args=ocp.args.StandardRestore(abstract_state),
                )
                start_step = restore_checkpoint_state(model, optimizer, restored)
                print(f"Resumed from checkpoint at step {start_step}")

                # Restore Grain iterator state if the companion file exists.
                grain_state_path = os.path.join(
                    cfg.ckpt_dir, str(restore_step), "grain_train_state.bin"
                )
                if os.path.exists(grain_state_path):
                    with open(grain_state_path, "rb") as fh:
                        restore_iter_state(train_iter, fh.read())
                    print("  → Grain data iterator restored from checkpoint")
                else:
                    print(
                        "  → no Grain iterator state found; data starts from epoch beginning"
                    )
            else:
                print("No checkpoint found to resume from, starting fresh")

        # ── Training loop ────────────────────────────────────────────────────────────

        max_steps = cfg.max_steps
        last_val_loss = float("inf")  # track for checkpointing metrics

        run = wandb.init(
            project="nano-gpt-jax",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "n_layer": cfg.model.n_layer,
                "n_head": cfg.model.n_head,
                "n_embd": cfg.model.n_embd,
                "block_size": cfg.model.block_size,
                "vocab_size": cfg.model.vocab_size,
                "batch_size": cfg.batch_size,
                "sequence_length": cfg.sequence_length,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "dtype_policy": cfg.apply_dtype_policy,
            },
        )

        t0 = time.time()
        for micro_step, batch in enumerate(
            train_iter, start=start_step * cfg.grad_acc_steps
        ):
            if micro_step >= max_steps * cfg.grad_acc_steps:
                break

            # Validation evaluation / Checkpointing
            if micro_step % (cfg.val_check_steps * cfg.grad_acc_steps) == 0:
                val_loss = last_val_loss
                if val_loader is not None:
                    t_val_start = time.time()
                    val_loss_accum = 0.0
                    val_steps = 0

                    # Each iter() call restarts from the beginning of the val split
                    # (num_epochs=1 in the val loader).
                    for val_batch in iter(val_loader):
                        if val_steps >= cfg.val_max_steps:
                            break
                        # Shard validation inputs along the data dimension
                        x_val_sharded = jax.device_put(
                            val_batch["x"], NamedSharding(mesh, P("data", None))
                        )
                        y_val_sharded = jax.device_put(
                            val_batch["y"], NamedSharding(mesh, P("data", None))
                        )
                        v_loss = val_step(model, x_val_sharded, y_val_sharded)
                        val_loss_accum += v_loss.item()
                        val_steps += 1

                    t_val_end = time.time()
                    val_dt = t_val_end - t_val_start

                    total_val_tokens = val_steps * cfg.sequence_length * cfg.batch_size
                    val_tokens_per_sec = (
                        total_val_tokens / val_dt if val_dt > 0 else 0.0
                    )

                    val_loss = (
                        val_loss_accum / val_steps if val_steps > 0 else float("inf")
                    )
                    last_val_loss = val_loss
                    print(
                        f"step {align_acc_step(micro_step, cfg.grad_acc_steps):4d} | validation loss {val_loss:.4f} | val_time {val_dt * 1000:.2f} ms | val_tokens/sec {val_tokens_per_sec:.2f}"
                    )
                    run.log(
                        {
                            "val_loss": val_loss,
                            "val_time_ms": val_dt * 1000,
                            "val_tokens_per_sec": val_tokens_per_sec,
                        },
                        step=align_acc_step(micro_step, cfg.grad_acc_steps),
                    )

                # ── Checkpoint (async) ───────────────────────────────────────────
                eff_step = align_acc_step(micro_step, cfg.grad_acc_steps)
                if eff_step > 0 and eff_step % cfg.ckpt_every_steps == 0:
                    ckpt_state = get_checkpoint_state(model, optimizer, eff_step)
                    ckpt_mngr.save(
                        eff_step,
                        args=ocp.args.StandardSave(ckpt_state),
                        metrics={"val_loss": val_loss},
                    )
                    print(f"  → checkpoint saved at step {eff_step}")

                    # Wait for the async Orbax write to finish so the step
                    # directory exists before we write the Grain state file.
                    ckpt_mngr.wait_until_finished()

                    # Persist Grain iterator state alongside the checkpoint.
                    grain_state_path = os.path.join(
                        cfg.ckpt_dir, str(eff_step), "grain_train_state.bin"
                    )
                    with open(grain_state_path, "wb") as fh:
                        fh.write(get_iter_state(train_iter))

                    # Upload to W&B as artifact
                    if not isinstance(wandb, MockWandb):
                        try:
                            artifact = wandb.Artifact(
                                f"checkpoint-step-{eff_step}", type="model"
                            )
                            ckpt_path = os.path.join(cfg.ckpt_dir, str(eff_step))
                            if os.path.isdir(ckpt_path):
                                artifact.add_dir(ckpt_path)
                                run.log_artifact(artifact)
                        except Exception as e:
                            print(f"  ⚠ W&B artifact upload failed: {e}")

                # Reset timer so validation time doesn't pollute training throughput
                t0 = time.time()

            # Shard training batch along the data mesh dimension
            x_sharded = jax.device_put(batch["x"], NamedSharding(mesh, P("data", None)))
            y_sharded = jax.device_put(batch["y"], NamedSharding(mesh, P("data", None)))
            loss = train_step(model, optimizer, x_sharded, y_sharded)
            loss.block_until_ready()  # ensure device work is done before timing

            if micro_step % cfg.grad_acc_steps == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    cfg.sequence_length * cfg.batch_size * cfg.grad_acc_steps / dt
                )
                total_tokens = (
                    micro_step
                    * cfg.grad_acc_steps
                    * cfg.sequence_length
                    * cfg.batch_size
                )
                run.log(
                    {
                        "loss": loss.item(),
                        "step_time_ms": dt * 1000,
                        "tokens_per_sec": tokens_per_sec,
                        "learning_rate": schedule(optimizer.step[...]).item(),
                        "total_tokens": total_tokens,
                    },
                    step=align_acc_step(micro_step, cfg.grad_acc_steps),
                )
                print(
                    f"step {align_acc_step(micro_step, cfg.grad_acc_steps):4d} | loss {loss:.4f} | lr: {schedule(optimizer.step[...]):.4f} | time {dt * 1000:.2f} ms | tokens/sec {tokens_per_sec:.2f} | tokens {total_tokens}"
                )
                t0 = time.time()

        # ── Cleanup ──────────────────────────────────────────────────────────────────
        ckpt_mngr.wait_until_finished()
        ckpt_mngr.close()
        run.finish()
