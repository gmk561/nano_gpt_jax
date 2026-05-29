import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx


class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig):
        self.config = config

        self.transformer = nnx.Dict({
            'wte': nnx.Embedding(config.vocab_size, config.n_embd),
            'wpe': nnx.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(0.1),
        })
        

