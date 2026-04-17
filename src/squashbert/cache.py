"""On-disk embedding caches: a numpy memmap plus a key->row mapping.

Two use cases:

- **Node cache.** ~2M keys (node CURIEs), fp16 embeddings. ~3 GB memmap.
- **Edge-type cache.** Thousands of keys. fp16 embeddings for the forward and reverse
  renderings, so 2 rows per edge type.

Layout on disk (a cache is a directory):
- `vectors.npy`: `np.memmap` of shape (N, D), dtype fp16.
- `keys.jsonl`: one key per line, `{"key": ..., "row": int}`. Keys are whatever the
  caller wants (strings for nodes; see `edge_type_key` for edges).
- `meta.json`: `{"dim": D, "dtype": "float16", "count": N}`.

Write once, read many. Not concurrent-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from .render import EdgeSpec


def edge_type_key(subject_cat: str, spec: EdgeSpec, object_cat: str, reverse: bool) -> str:
    """Canonical string key for one direction of an edge type."""
    asp = spec.object_aspect or ""
    dir_ = spec.object_direction or ""
    d = "R" if reverse else "F"
    return f"{subject_cat}|{spec.predicate}|{asp}|{dir_}|{object_cat}|{d}"


@dataclass
class EmbeddingCache:
    dir: Path
    vectors: np.memmap
    key_to_row: dict[str, int]
    dim: int

    @classmethod
    def create(
        cls, dir: str | Path, keys: list[str], dim: int, dtype: str = "float16"
    ) -> "EmbeddingCache":
        """Allocate an empty cache on disk. Caller fills `vectors[row]` then calls `flush()`."""
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        n = len(keys)
        vectors = np.memmap(d / "vectors.npy", dtype=dtype, mode="w+", shape=(n, dim))
        key_to_row = {k: i for i, k in enumerate(keys)}
        with open(d / "keys.jsonl", "wb") as f:
            for i, k in enumerate(keys):
                f.write(orjson.dumps({"key": k, "row": i}))
                f.write(b"\n")
        (d / "meta.json").write_text(json.dumps({"dim": dim, "dtype": dtype, "count": n}))
        return cls(dir=d, vectors=vectors, key_to_row=key_to_row, dim=dim)

    @classmethod
    def load(cls, dir: str | Path) -> "EmbeddingCache":
        d = Path(dir)
        meta = json.loads((d / "meta.json").read_text())
        vectors = np.memmap(
            d / "vectors.npy", dtype=meta["dtype"], mode="r", shape=(meta["count"], meta["dim"])
        )
        key_to_row: dict[str, int] = {}
        with open(d / "keys.jsonl", "rb") as f:
            for line in f:
                rec = orjson.loads(line)
                key_to_row[rec["key"]] = rec["row"]
        return cls(dir=d, vectors=vectors, key_to_row=key_to_row, dim=meta["dim"])

    def flush(self) -> None:
        self.vectors.flush()

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.key_to_row[key]]

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_row

    def __len__(self) -> int:
        return len(self.key_to_row)
