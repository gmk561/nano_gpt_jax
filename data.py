"""
Data loading for nano-GPT JAX training.

Provides Grain-based data source implementations and factory functions for
training and validation data loaders.

All loaders use :class:`grain.ShardByJaxProcess` so that multi-host JAX
training automatically assigns non-overlapping data subsets to each host
process.  In a single-process setup (the common case), this is equivalent to
no sharding.

Batching is done per local process::

    local_batch_size = cfg.device_batch_size * jax.local_device_count()

The caller is responsible for placing the resulting numpy batches onto devices,
e.g.::

    for batch in train_iter:
        x = jax.device_put(batch["x"], NamedSharding(mesh, P("data", None)))
        y = jax.device_put(batch["y"], NamedSharding(mesh, P("data", None)))

Checkpointing
-------------
Grain state lives on the **iterator**, not the loader.  Keep a reference to the
iterator returned by ``iter(loader)`` and use :func:`get_iter_state` /
:func:`restore_iter_state` to save and resume from an exact position::

    train_iter = iter(train_loader)
    ...                                   # training loop
    state = get_iter_state(train_iter)    # bytes — save to disk
    ...
    restore_iter_state(train_iter, state) # seek to saved position
"""

from __future__ import annotations

import bisect
import os
from typing import Any

import jax
import numpy as np
import tiktoken
from ml_collections import ConfigDict

import grain.python as grain

# Grain uses absl.flags internally.  Ensure flags are marked as parsed now so
# that background worker threads don't raise UnparsedFlagAccessError when they
# first try to read Grain's profiling flag.
from absl import flags as _absl_flags

if not _absl_flags.FLAGS.is_parsed():
    _absl_flags.FLAGS.mark_as_parsed()


GPT2_VOCAB_SIZE: int = 50304

_EDU_FINEWEB_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")



class EduFinewebShardSource:
    """Grain-compatible data source over pre-tokenized EduFineweb ``.npy`` shards.

    Shards are discovered by matching the pattern
    ``edufineweb_{split}_*.npy`` inside *data_dir* and sorted by filename.

    Each item ``__getitem__(idx)`` returns a dict::

        {"x": int32[sequence_length], "y": int32[sequence_length]}

    Shards are loaded lazily on first access and cached per instance (i.e. per
    Grain worker process), so each worker keeps at most one shard in memory at
    a time.

    Parameters
    ----------
    sequence_length:
        Number of tokens per training / validation window.
    split:
        ``"train"`` or ``"val"``.
    data_dir:
        Directory containing the ``.npy`` shard files.  Defaults to the
        ``edu_fineweb10B/`` folder next to this module.
    """

    def __init__(
        self,
        sequence_length: int,
        split: str = "train",
        data_dir: str = _EDU_FINEWEB_DATA_DIR,
    ) -> None:
        self._seq_len = sequence_length
        self._split = split
        self._data_dir = data_dir

        shard_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith(f"edufineweb_{split}_") and f.endswith(".npy")
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No .npy shards for split='{split}' in {data_dir}. "
                "Run `python fineweb.py` first."
            )
        self._shard_paths = shard_paths

        # Read each shard's token count from the .npy header only (no full load).
        token_counts = [int(np.load(p, mmap_mode="r").shape[0]) for p in shard_paths]

        # Non-overlapping windows per shard (need seq_len+1 consecutive tokens).
        seqs_per_shard = [max(0, (n - 1) // sequence_length) for n in token_counts]

        # Prefix sums for fast global-index → (shard_idx, local_idx) mapping.
        self._cumulative: list[int] = [0]
        for s in seqs_per_shard:
            self._cumulative.append(self._cumulative[-1] + s)

        # Per-instance shard cache (one shard at a time, populated lazily).
        self._cached_shard_idx: int | None = None
        self._cached_tokens: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"EduFinewebShardSource(sequence_length={self._seq_len}, "
            f"split={self._split!r}, data_dir={self._data_dir!r})"
        )

    # ── Grain RandomAccessDataSource protocol ──────────────────────────────────

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # Map global index → (shard_idx, local_idx).
        shard_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[shard_idx]

        if self._cached_shard_idx != shard_idx:
            # mmap_mode='r' memory-maps the file: the OS pages in only the
            # regions actually accessed, keeping RAM usage to a tiny fraction
            # of the full shard (~10 GB as uint16 on disk).  Critically, we
            # do NOT convert the whole array to int32 here — that would
            # materialise 20 GB per worker process and trigger the OOM killer.
            self._cached_tokens = np.load(
                self._shard_paths[shard_idx], mmap_mode="r"
            )
            self._cached_shard_idx = shard_idx

        start = local_idx * self._seq_len
        # Slice first (tiny copy), then cast to int32 — only seq_len+1 elements.
        buf = self._cached_tokens[start : start + self._seq_len + 1].astype(np.int32)
        return {"x": buf[:-1].copy(), "y": buf[1:].copy()}


# ── input.txt data source ──────────────────────────────────────────────────────


class InputTxtSource:
    """Grain-compatible data source for the single-file ``input.txt`` dataset.

    Tokenises the file with the GPT-2 tiktoken encoder at construction time and
    keeps the full token sequence in memory.

    Each item ``__getitem__(idx)`` returns a dict::

        {"x": int32[sequence_length], "y": int32[sequence_length]}

    Parameters
    ----------
    sequence_length:
        Number of tokens per window.
    path:
        Path to ``input.txt``.  Defaults to the file next to this module.
    """

    def __init__(
        self,
        sequence_length: int,
        path: str = os.path.join(os.path.dirname(__file__), "input.txt"),
    ) -> None:
        self._seq_len = sequence_length
        self._path = path
        enc = tiktoken.get_encoding("gpt2")
        with open(path, "r", encoding="utf-8") as fh:
            self._tokens = np.array(enc.encode(fh.read()), dtype=np.int32)

    def __repr__(self) -> str:
        return f"InputTxtSource(sequence_length={self._seq_len}, path={self._path!r})"

    def __len__(self) -> int:
        return max(0, (len(self._tokens) - 1) // self._seq_len)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start = idx * self._seq_len
        buf = self._tokens[start : start + self._seq_len + 1]
        return {"x": buf[:-1].copy(), "y": buf[1:].copy()}




# ── Internal helpers ───────────────────────────────────────────────────────────


def _local_batch_size(cfg: "ConfigDict") -> int:
    """Per-process batch size: ``device_batch_size × local_device_count``.

    In a single-process run this equals ``cfg.batch_size``.  In a multi-host
    setup each process loads only its share of the global batch.
    """
    return cfg.device_batch_size * jax.local_device_count()


def _make_source(cfg: "ConfigDict") -> EduFinewebShardSource | InputTxtSource:
    """Instantiate the appropriate training data source from ``cfg.dataset``."""
    if cfg.dataset == "edu_fineweb":
        return EduFinewebShardSource(sequence_length=cfg.sequence_length)
    if cfg.dataset == "input_txt":
        return InputTxtSource(sequence_length=cfg.sequence_length)
    raise ValueError(
        f"Unknown dataset '{cfg.dataset}'. Expected 'edu_fineweb' or 'input_txt'."
    )


# ── Loader factories ───────────────────────────────────────────────────────────


def create_train_loader(
    cfg: "ConfigDict",
    *,
    seed: int = 0,
    worker_count: int | None = None,
) -> grain.DataLoader:
    """Build a Grain :class:`~grain.python.DataLoader` for training.

    The loader shuffles data deterministically and iterates indefinitely
    (``num_epochs=None``); the training loop controls when to stop.
    Each JAX process receives a unique, non-overlapping shard of the data
    via :class:`~grain.python.ShardByJaxProcess`.

    Parameters
    ----------
    cfg:
        Training configuration.  Must contain ``dataset``,
        ``device_batch_size``, and ``sequence_length``.
    seed:
        Shuffle seed for the :class:`~grain.python.IndexSampler`.
    worker_count:
        Number of background worker processes for data prefetching.
        Defaults to ``min(4, cpu_count // 2)`` for EduFineweb (I/O bound)
        and ``0`` for ``input_txt`` (data fits in memory, no I/O).

    Returns
    -------
    A :class:`~grain.python.DataLoader` whose iterator yields
    ``{"x": int32[local_batch, seq_len], "y": int32[local_batch, seq_len]}``
    dicts.

    Note
    ----
    Call ``iter()`` on the returned loader to obtain a
    :class:`~grain.python.DataLoaderIterator` from which you can call
    :func:`get_iter_state` / :func:`restore_iter_state` for checkpointing.
    """
    source = _make_source(cfg)
    local_bs = _local_batch_size(cfg)

    if worker_count is None:
        worker_count = (
            0
            if cfg.dataset == "input_txt"
            else min(4, max(1, (os.cpu_count() or 2) // 2))
        )

    sampler = grain.IndexSampler(
        len(source),
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=False,
        num_epochs=None,
        seed=seed,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=local_bs, drop_remainder=True)],
        worker_count=worker_count,
        worker_buffer_size=2,
    )


def create_val_loader(cfg: "ConfigDict") -> grain.DataLoader | None:
    """Build a Grain :class:`~grain.python.DataLoader` for validation.

    Returns ``None`` for the ``"input_txt"`` dataset (no dedicated val split).
    Each call to ``iter(loader)`` performs one full pass through the per-process
    validation shard (``num_epochs=1``).

    Parameters
    ----------
    cfg:
        Training configuration.

    Returns
    -------
    A :class:`~grain.python.DataLoader` or ``None``.
    """
    if cfg.dataset == "input_txt":
        return None
    source = EduFinewebShardSource(sequence_length=cfg.sequence_length, split="val")
    local_bs = _local_batch_size(cfg)

    sampler = grain.IndexSampler(
        len(source),
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=False,
        num_epochs=1,
        seed=0,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=local_bs, drop_remainder=True)],
        worker_count=0,
    )




# ── Iterator checkpoint helpers ────────────────────────────────────────────────


def get_iter_state(it: grain.DataLoaderIterator) -> bytes:
    """Return a snapshot of the iterator's current position.

    The returned value is a byte string (JSON-encoded internally by Grain).
    Write it directly to a binary file for checkpointing::

        state = get_iter_state(train_iter)
        with open(path, "wb") as fh:
            fh.write(state)

    Parameters
    ----------
    it:
        The :class:`~grain.python.DataLoaderIterator` to snapshot.  Call this
        *after* consuming the batch for the step you want to checkpoint so that
        restoring the state will resume from the *next* batch.
    """
    return it.get_state()


def restore_iter_state(it: grain.DataLoaderIterator, state: bytes) -> None:
    """Seek an iterator to a previously saved position.

    Parameters
    ----------
    it:
        A freshly created :class:`~grain.python.DataLoaderIterator` (i.e.
        the result of ``iter(loader)``).  Must not have produced any elements
        yet — call this immediately after creating the iterator.
    state:
        Bytes previously returned by :func:`get_iter_state`.
    """
    it.set_state(state)
