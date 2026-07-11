"""
Downloads and evaluates HellaSwag in Python (JAX/Flax version).
https://github.com/rowanz/hellaswag

Code from:
https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he",
 "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3,
 "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
             "is holding a rubik's cube.", "starts pulling up roofing on a roof."],
 "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
ctx: The full context string (ctx_a + " " + ctx_b when ctx_b is nonempty)
endings: a list of 4 endings; correct index given by label (0-3)
split: train, val, or test
split_type: indomain if activity label seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import argparse
import numpy as np
import requests
import tiktoken
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax import nnx

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswag_urls = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


# ── Download helpers ───────────────────────────────────────────────────────────


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Stream-download a file with a tqdm progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as fh, tqdm(
        desc=os.path.basename(fname),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))


def download(split: str) -> None:
    """Download a HellaSwag split into DATA_CACHE_DIR if not already present."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    url = hellaswag_urls[split]
    dest = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(dest):
        print(f"Downloading {url} -> {dest} ...")
        download_file(url, dest)


# ── Data helpers ───────────────────────────────────────────────────────────────


def render_example(example: dict) -> tuple:
    """
    Tokenise a HellaSwag example into padded numpy arrays.

    Returns
    -------
    data   : dict with raw token lists (for debugging)
    tokens : int32 array of shape (4, max_len)  - context + each ending
    mask   : int32 array of shape (4, max_len)  - 1 only over the ending tokens
    label  : int, index of the correct ending
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    data = {"label": label, "ctx_tokens": ctx_tokens, "ending_tokens": []}

    tok_rows, mask_rows = [], []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepend space as GPT-2 tokenizer expects
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(r) for r in tok_rows)
    tokens = np.zeros((4, max_len), dtype=np.int32)
    mask = np.zeros((4, max_len), dtype=np.int32)
    for i, (tr, mr) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tr)] = tr
        mask[i, : len(mr)] = mr

    return data, tokens, mask, label


def iterate_examples(split: str):
    """Yield parsed HellaSwag examples for *split* (downloading if needed)."""
    download(split)
    path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(path, "r") as fh:
        for line in fh:
            yield json.loads(line)


# ── Evaluation ─────────────────────────────────────────────────────────────────


def evaluate(
    model: nnx.Module, split: str = "val", max_examples: int | None = None
) -> dict:
    """
    Evaluate *model* on HellaSwag using the completion-style approach.

    For each example the 4 candidate completions are scored by measuring
    cross-entropy loss over the completion tokens only (mask == 1).

    Parameters
    ----------
    model        : a Flax NNX model with a forward pass ``model(tokens) -> logits``
                   where logits has shape (B, T, vocab_size).
    split        : "train", "val", or "test"
    max_examples : if set, stop after this many examples (useful for quick checks)

    Returns
    -------
    dict with keys acc, acc_norm, num_total
    """
    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    for example in iterate_examples(split):
        data, tokens_np, mask_np, label = render_example(example)

        tokens = jnp.array(tokens_np)  # (4, T)
        mask = jnp.array(mask_np)  # (4, T)

        # Forward pass: run all 4 candidates together
        logits = model(tokens)  # (4, T, vocab_size)

        # Shift for autoregressive loss
        shift_logits = logits[:, :-1, :]  # (4, T-1, V)
        shift_tokens = tokens[:, 1:]  # (4, T-1)
        shift_mask = mask[:, 1:]  # (4, T-1)

        # Per-token cross-entropy (no reduction)
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_tokens = shift_tokens.reshape(-1)
        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
        per_token_loss = -log_probs[jnp.arange(flat_tokens.shape[0]), flat_tokens]
        per_token_loss = per_token_loss.reshape(4, -1)  # (4, T-1)

        # Restrict to completion region
        masked_loss = per_token_loss * shift_mask
        sum_loss = masked_loss.sum(axis=1)  # (4,)
        avg_loss = sum_loss / jnp.maximum(shift_mask.sum(axis=1), 1.0)  # (4,)

        pred = int(jnp.argmin(sum_loss))
        pred_norm = int(jnp.argmin(avg_loss))

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        print(
            f"{num_total:5d} | "
            f"acc: {num_correct/num_total:.4f} | "
            f"acc_norm: {num_correct_norm/num_total:.4f}",
            end="\r",
        )

        # Debug: pretty-print the first few examples
        if num_total <= 5:
            print()
            print("---")
            print(f"Context: {example['ctx']}")
            for i, end in enumerate(example["endings"]):
                marker = "v" if i == label else " "
                print(f"  {marker} [{i}] (loss={avg_loss[i].item():.4f}) {end}")
            print(f"  -> predicted: {pred_norm}  actual: {label}")

        if max_examples is not None and num_total >= max_examples:
            break

    print()  # newline after \r progress
    results = {
        "acc": num_correct / num_total,
        "acc_norm": num_correct_norm / num_total,
        "num_total": num_total,
    }
    print(
        f"HellaSwag {split} | "
        f"n={num_total} | "
        f"acc={results['acc']:.4f} | "
        f"acc_norm={results['acc_norm']:.4f}"
    )
    return results


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a JAX GPT on HellaSwag")
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Stop after this many examples (omit for full eval)",
    )
    args = parser.parse_args()

    # Import here to avoid circular issues when used as a library
    from config import get_config
    from model import GPT
    from data import GPT2_VOCAB_SIZE

    cfg = get_config()
    cfg.model.vocab_size = GPT2_VOCAB_SIZE

    mesh = jax.make_mesh((cfg.num_devices, 1), ("data", "model"))
    rngs = nnx.Rngs(0)

    with jax.set_mesh(mesh):
        model = GPT(cfg.model, rngs=rngs)
        evaluate(model, split=args.split, max_examples=args.max_examples)
