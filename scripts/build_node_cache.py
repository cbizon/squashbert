"""Scan nodes.jsonl and write a fp16 [CLS]-pooled SAPBERT embedding cache.

Usage:
    uv run python scripts/build_node_cache.py \\
        --nodes path/to/nodes.jsonl --out caches/nodes [--batch-size 256]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from squashbert.cache import EmbeddingCache
from squashbert.embed import EMBED_DIM, Embedder
from squashbert.kgx import load_nodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=32)
    args = ap.parse_args()

    print(f"Loading nodes from {args.nodes}...")
    nodes = load_nodes(args.nodes)
    print(f"  {len(nodes):,} nodes retained")

    ids = list(nodes.keys())
    cache = EmbeddingCache.create(args.out, keys=ids, dim=EMBED_DIM, dtype="float16")

    print("Loading SAPBERT...")
    embedder = Embedder()

    print(f"Embedding on {embedder.device}...")
    bs = args.batch_size
    for i in tqdm(range(0, len(ids), bs)):
        batch_ids = ids[i : i + bs]
        names = [nodes[nid].name for nid in batch_ids]
        vecs = embedder.embed_cls(names, batch_size=bs, max_length=args.max_length)
        cache.vectors[i : i + len(batch_ids)] = vecs.astype(np.float16)
    cache.flush()
    print(f"Wrote {len(ids):,} vectors to {args.out}")


if __name__ == "__main__":
    main()
