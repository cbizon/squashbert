"""Enumerate distinct edge types from edges.jsonl, embed each in both directions,
and write the edge-type list (pickled) alongside a fp16 embedding cache.

Usage:
    uv run python scripts/build_edge_cache.py \\
        --nodes path/to/nodes.jsonl --edges path/to/edges.jsonl --out caches/edges [--pooling cls]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from squashbert.cache import EmbeddingCache, edge_type_key
from squashbert.embed import EMBED_DIM, Embedder
from squashbert.kgx import collect_edge_types, load_nodes
from squashbert.render import render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--edges", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--pooling", choices=["cls", "mean"], default="cls",
        help="Pooling strategy: cls (default, matches SAPBERT training) or mean",
    )
    args = ap.parse_args()

    print(f"Loading nodes from {args.nodes}...")
    nodes = load_nodes(args.nodes)
    print(f"  {len(nodes):,} nodes")

    print(f"Scanning edges from {args.edges} (this reads the whole file)...")
    edge_types, skips = collect_edge_types(args.edges, nodes)
    print(f"  {len(edge_types):,} distinct edge types; skipped: {skips}")

    # Persist the edge-type list.
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "edge_types.pkl", "wb") as f:
        pickle.dump(edge_types, f)

    # Build forward + reverse renderings, deduped by cache key.
    keys: list[str] = []
    phrases: list[str] = []
    for et in edge_types:
        for rev in (False, True):
            key = edge_type_key(et.subject_category, et.spec, et.object_category, reverse=rev)
            keys.append(key)
            phrases.append(render(et.spec, reverse=rev))

    cache = EmbeddingCache.create(args.out, keys=keys, dim=EMBED_DIM, dtype="float16")

    print("Loading SAPBERT...")
    embedder = Embedder()

    embed_fn = embedder.embed_cls if args.pooling == "cls" else embedder.embed_sentence
    print(f"Embedding {len(phrases):,} edge renderings on {embedder.device} ({args.pooling} pooling)...")
    bs = args.batch_size
    for i in tqdm(range(0, len(phrases), bs)):
        vecs = embed_fn(phrases[i : i + bs], batch_size=bs, max_length=16)
        cache.vectors[i : i + len(vecs)] = vecs.astype(np.float16)
    cache.flush()
    print(f"Wrote {len(keys):,} edge vectors ({args.pooling} pooling) to {args.out}")


if __name__ == "__main__":
    main()
