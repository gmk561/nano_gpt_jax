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

if args.smoke_test:
    shard_size = int(1e5)  # tiny shard for smoke test

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# in smoke-test mode, take only 20 documents
if args.smoke_test:
    print("=== SMOKE TEST MODE: processing 20 documents only ===")
    fw = fw.take(20)

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

    def process(token_iterator):
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in token_iterator:
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None

                # stop early if --num-shards limit reached
                if args.num_shards is not None and shard_index >= args.num_shards:
                    print(f"Reached --num-shards={args.num_shards} limit, stopping.")
                    return

                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

    if args.smoke_test:
        # smoke-test: no multiprocessing at all — plain sequential loop, no SIGSEGV
        process(map(tokenize, fw))
    else:
        # full mode: parallel tokenization with a process pool
        with mp.Pool(nprocs) as pool:
            process(pool.imap(tokenize, fw, chunksize=16))

