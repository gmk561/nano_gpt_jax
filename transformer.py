from pydantic._internal import _generate_schema
from IPython.core import interactiveshell
import os
import time
import tiktoken
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
from flax import nnx
from typing import Callable
from ml_collections import ConfigDict


print("jax.device_count():", jax.device_count())


# Load environment variables from local .env if it exists
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

if os.environ.get("WANDB_MODE") == "disabled":

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

    wandb = MockWandb()
else:
    import wandb


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.sequence_length = 1024
    cfg.apply_dtype_policy = True
    cfg.num_devices = jax.device_count()
    cfg.device_batch_size = 16
    cfg.batch_size = cfg.device_batch_size * cfg.num_devices
    cfg.gpt_batch_size = 524288
    cfg.grad_acc_steps = cfg.gpt_batch_size // cfg.batch_size
    cfg.dataset = "edu_fineweb"  # "input_txt" or "edu_fineweb"

    # GPT model config
    cfg.model = ConfigDict()
    cfg.model.block_size = cfg.sequence_length
    cfg.model.vocab_size = 65
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
    cfg.batch_size = 16
    cfg.gpt_batch_size = 16
    cfg.grad_acc_steps = 1
    # cfg.dataset = "input_txt"

    return cfg


cfg = get_cpu_test_config()


class MLP(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config
        self.linear_1 = nnx.Linear(
            config.n_embd, 4 * config.n_embd, rngs=rngs, dtype=config.compute_dtype
        )
        self.linear_2 = nnx.Linear(
            4 * config.n_embd, config.n_embd, rngs=rngs, dtype=config.compute_dtype
        )

    def __call__(self, x: jnp.ndarray):
        return self.linear_2(nnx.gelu(self.linear_1(x)))


class Block(nnx.Module):
    def __init__(self, config: ConfigDict, rngs: nnx.Rngs):
        self.config = config
        self.mha = nnx.MultiHeadAttention(
            num_heads=config.n_head,
            in_features=config.n_embd,
            qkv_features=config.n_embd,  # total dim; Flax splits by num_heads internally
            rngs=rngs,
            decode=False,
            dtype=config.compute_dtype,
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
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embd,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.compute_dtype,
        )
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embd,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.compute_dtype,
        )
        self.blocks = nnx.List(
            [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        )
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)
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

        def _apply(module, parent=None, _attr_name=None):
            # # Mixed precision: use bfloat16 for faster computation, weights remain float32, activations are bfloat16
            # # don't apply to LayerNorm as it sums many values, this can lead to overflow or underflow.
            # if isinstance(module, (nnx.Linear, nnx.MultiHeadAttention, nnx.Embed)):
            #     module.dtype = jnp.bfloat16

            if isinstance(module, nnx.Linear):
                stddev = (
                    0.02
                    if not _is_residual_output(module, parent, _attr_name)
                    else 0.02 * residual_scale
                )
                module.kernel.value = nnx.initializers.normal(stddev=stddev)(
                    rngs.params(), module.kernel[...].shape
                )
                if module.use_bias:
                    module.bias.value = jnp.zeros(module.bias[...].shape)
            elif isinstance(module, nnx.Embed):
                module.embedding.value = nnx.initializers.normal(stddev=0.02)(
                    rngs.params(), module.embedding[...].shape
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

    def __call__(self, x: jnp.ndarray):
        B, T = x.shape
        mask = nnx.make_causal_mask(x)
        pos = jnp.arange(T)
        x = self.wte(x) + self.wpe(pos)  # (B, T, n_embd)
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


class DataLoader:
    def __init__(self, batch_size: int, sequence_length: int):

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        # self.enc = CharTokenizer(text)
        self.enc = tiktoken.get_encoding("gpt2")
        # self.vocab_size = self.enc.vocab_size
        self.vocab_size = self.enc.n_vocab
        self.tokens = jnp.array(self.enc.encode(text))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.idx = 0

    def __next__(self):
        if self.idx + self.sequence_length * self.batch_size >= len(self.tokens):
            self.idx = 0

        x = self.tokens[self.idx : self.idx + self.sequence_length * self.batch_size]
        y = self.tokens[
            self.idx + 1 : self.idx + self.sequence_length * self.batch_size + 1
        ]

        self.idx += self.sequence_length * self.batch_size

        return x.reshape(self.batch_size, self.sequence_length), y.reshape(
            self.batch_size, self.sequence_length
        )

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return len(self.tokens) // (self.sequence_length * self.batch_size)


class EduFinewebDataLoader:
    """Streams pre-tokenized .npy shards produced by fineweb.py.

    Shards live in ``data_dir`` with names like
    ``edufineweb_train_000001.npy`` / ``edufineweb_val_000000.npy``.

    The loader cycles through all shards for the given *split*, advancing
    the internal pointer exactly like the simple DataLoader above.
    """

    DATA_DIR = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")

    def __init__(self, batch_size: int, sequence_length: int, split: str = "train"):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.split = split

        # GPT-2 vocab size (fixed for this tokenizer)
        self.vocab_size = tiktoken.get_encoding("gpt2").n_vocab

        # Discover shard files for this split, sorted by index
        self.shard_paths = sorted(
            [
                os.path.join(self.DATA_DIR, f)
                for f in os.listdir(self.DATA_DIR)
                if f.startswith(f"edufineweb_{split}_") and f.endswith(".npy")
            ]
        )
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No .npy shards found for split='{split}' in {self.DATA_DIR}. "
                f"Run `python fineweb.py` first."
            )

        self.shard_idx = 0
        self.idx = 0
        self._load_shard(0)

    def _load_shard(self, shard_idx: int) -> None:
        """Load a single shard into memory as a jnp int32 array."""
        self.shard_idx = shard_idx % len(self.shard_paths)
        self.tokens = jnp.array(
            np.load(self.shard_paths[self.shard_idx]).astype(np.int32)
        )
        self.idx = 0

    def __next__(self):
        B, T = self.batch_size, self.sequence_length
        needed = B * T + 1  # +1 for the target shift

        # If the current shard is exhausted, advance to the next one
        if self.idx + needed > len(self.tokens):
            self.shard_idx += 1
            self._load_shard(self.shard_idx)

        buf = self.tokens[self.idx : self.idx + needed]
        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)
        self.idx += B * T
        return x, y

    def __iter__(self):
        self._load_shard(0)
        return self

    def __len__(self):
        """Approximate total steps across all shards (each shard counted once)."""
        total_tokens = sum(
            os.path.getsize(p) // 2  # uint16 → 2 bytes per token
            for p in self.shard_paths
        )
        return total_tokens // (self.sequence_length * self.batch_size)


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


# ── Training setup ──────────────────────────────────────────────────────────

dataset = DataLoader(batch_size=cfg.batch_size, sequence_length=cfg.sequence_length)

cfg.model.vocab_size = (
    dataset.vocab_size
)  # derived from the actual characters in the text

rngs = nnx.Rngs(0)
model = GPT(cfg.model, rngs=rngs)

if cfg.apply_dtype_policy:
    apply_dtype_policy(model, cfg.model)


def dtype_report(model: nnx.Module):
    for path, module in model.iter_modules():
        for attr in ("kernel", "embedding", "scale", "bias"):
            param = getattr(module, attr, None)
            if param is not None and hasattr(param, "value"):
                print(f"{path} {attr} {param[...].dtype}")


dtype_report(model)


learning_rate = 6e-4
warmup_steps = 10
decay_steps = 50 - warmup_steps

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=learning_rate,
    warmup_steps=warmup_steps,
    decay_steps=decay_steps,
    end_value=learning_rate * 0.1,
)


tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1),
)
if cfg.grad_acc_steps > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=cfg.grad_acc_steps)

optimizer = nnx.Optimizer(
    model,
    tx,
    wrt=nnx.Param,
)


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


def align_acc_step(step: int, gradient_acc_steps: int) -> int:
    return step // gradient_acc_steps


if __name__ == "__main__":
    # ── Training setup ──────────────────────────────────────────────────────────

    if cfg.dataset == "edu_fineweb":
        dataset = EduFinewebDataLoader(
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            split="train",
        )
    else:
        dataset = DataLoader(
            batch_size=cfg.batch_size, sequence_length=cfg.sequence_length
        )

    cfg.model.vocab_size = (
        dataset.vocab_size
    )  # derived from the actual characters in the text

    rngs = nnx.Rngs(0)
    model = GPT(cfg.model, rngs=rngs)

    if cfg.apply_dtype_policy:
        apply_dtype_policy(model, cfg.model)

    dtype_report(model)

    learning_rate = 6e-4
    warmup_steps = 10
    decay_steps = 50 - warmup_steps

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=learning_rate * 0.1,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1),
    )
    if cfg.grad_acc_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=cfg.grad_acc_steps)

    optimizer = nnx.Optimizer(
        model,
        tx,
        wrt=nnx.Param,
    )

    # ── Training loop ────────────────────────────────────────────────────────────

    max_steps = 50

    wandb.init(
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

    for step, (x, y) in enumerate(dataset):
        if step >= max_steps:
            break
        t0 = time.time()
        loss = train_step(model, optimizer, x, y)
        loss.block_until_ready()
        dt = time.time() - t0
        tokens_per_sec = dataset.sequence_length * dataset.batch_size / dt

        if step % cfg.grad_acc_steps == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "step_time_ms": dt * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "learning_rate": schedule(optimizer.step[...]).item(),
                },
                step=align_acc_step(step, cfg.grad_acc_steps),
            )
            print(
                f"step {align_acc_step(step, cfg.grad_acc_steps):4d} | loss {loss:.4f} | lr: {schedule(optimizer.step[...]):.4f} | time {dt * 1000:.2f} ms | tokens/sec {tokens_per_sec:.2f}"
            )

    wandb.finish()
