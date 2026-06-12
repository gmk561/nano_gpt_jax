"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".

For a quick smoke test (only 20 documents, no multiprocessing):
$ python fineweb.py --smoke-test

To download only the first N shards:
$ python fineweb.py --num-shards 1
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu dataset")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Process only 20 documents for a quick sanity check.",
)
parser.add_argument(
    "--num-shards",
    type=int,
    default=None,
    help="Stop after writing this many shards (default: write all shards).",
)
args = parser.parse_args()

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

val_shards = 2 if args.smoke_test else 1
if args.smoke_test:
    shard_size = 1024 * 16 + 1  # tiny shard (B * T + 1) to accommodate target shift
    total_shards = 4  # 2 val + 2 train
else:
    total_shards = args.num_shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# in smoke-test mode, clean old files in DATA_CACHE_DIR to avoid stale shards
if args.smoke_test:
    print(
        f"=== SMOKE TEST MODE: generating {val_shards} val and {total_shards - val_shards} train shards ==="
    )
    for f in os.listdir(DATA_CACHE_DIR):
        if f.startswith("edufineweb_") and f.endswith(".npy"):
            os.remove(os.path.join(DATA_CACHE_DIR, f))

# download the dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True
)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Number of processes used {nprocs}")

    def process(token_iterator):
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = tqdm(
            total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
        )
        for tokens in token_iterator:
            offset = 0
            while offset < len(tokens):
                space_left = shard_size - token_count
                chunk_size = min(space_left, len(tokens) - offset)
                all_tokens_np[token_count : token_count + chunk_size] = tokens[
                    offset : offset + chunk_size
                ]
                token_count += chunk_size
                offset += chunk_size
                progress_bar.update(chunk_size)

                if token_count == shard_size:
                    split = "val" if shard_index < val_shards else "train"
                    filename = os.path.join(
                        DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                    )
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar.close()

                    if total_shards is not None and shard_index >= total_shards:
                        print(f"Reached limit of {total_shards} shards, stopping.")
                        return

                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                    token_count = 0

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index < val_shards else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])
        progress_bar.close()

    if args.smoke_test:
        # smoke-test: no multiprocessing at all — plain sequential loop, no SIGSEGV
        process(map(tokenize, fw))
    else:
        # full mode: parallel tokenization with a process pool
        with mp.Pool(nprocs) as pool:
            process(pool.imap(tokenize, fw, chunksize=16))
