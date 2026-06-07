from flax.nnx import variablelib
from ast import mod
import time
import tiktoken
import jax
import jax.numpy as jnp
import optax
from flax import nnx


sequence_length = 32


class GPTConfig:
    block_size: int = 32
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.linear_1 = nnx.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.linear_2 = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        return self.linear_2(nnx.gelu(self.linear_1(x)))


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.mha = nnx.MultiHeadAttention(
            num_heads=config.n_head,
            in_features=config.n_embd,
            qkv_features=config.n_embd,  # total dim; Flax splits by num_heads internally
            rngs=rngs,
            decode=False,
        )
        self.mlp = MLP(config, rngs=rngs)
        self.layernorm_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):
        x = x + self.mha(self.layernorm_1(x), mask=mask)
        x = x + self.mlp(self.layernorm_2(x))

        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embd,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(stddev=0.02),
        )
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embd,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(stddev=0.02),
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

        def _apply(module, parent=None, attr_name=None):
            if isinstance(module, nnx.Linear):
                stddev = (
                    0.02
                    if not _is_residual_output(module, parent, attr_name)
                    else 0.02 * residual_scale
                )
                module.kernel.value = nnx.initializers.normal(stddev=stddev)(
                    rngs.params(), module.kernel.value.shape
                )
                if module.use_bias:
                    module.bias.value = jnp.zeros(module.bias.value.shape)
            elif isinstance(module, nnx.Embed):
                module.embedding.value = nnx.initializers.normal(stddev=0.02)(
                    rngs.params(), module.embedding.value.shape
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
        x = self.ln_f(x)
        # weight tying: reuse wte embedding matrix as output projection
        logits = x @ self.wte.embedding.value.T  # (B, T, vocab_size)
        return logits


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


# ── Training setup ──────────────────────────────────────────────────────────

dataset = DataLoader(batch_size=4, sequence_length=sequence_length)

config = GPTConfig()
config.block_size = sequence_length
config.vocab_size = dataset.vocab_size  # derived from the actual characters in the text
rngs = nnx.Rngs(0)
model = GPT(config, rngs=rngs)

learning_rate = 1e-3

optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)


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


# ── Training loop ────────────────────────────────────────────────────────────

max_steps = 50

# nnx.display(model)

x, y = dataset.__next__()

for step, (x0, y0) in enumerate(dataset):
    if step >= max_steps:
        break
    t0 = time.time()
    loss = train_step(model, optimizer, x, y)
    dt = time.time() - t0
    print(f"step {step:4d} | loss {loss:.4f} | time {dt * 1000:.2f} ms")
