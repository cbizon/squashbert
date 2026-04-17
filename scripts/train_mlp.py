"""Train one hop-count MLP until plateau. Requires node and edge caches built first.

Usage:
    uv run python scripts/train_mlp.py --hops 2 \\
        --nodes path/to/nodes.jsonl \\
        --node-cache caches/nodes \\
        --edge-cache caches/edges \\
        --out checkpoints/hop2
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from squashbert.cache import EmbeddingCache
from squashbert.embed import Embedder
from squashbert.kgx import build_category_index, load_nodes
from squashbert.sampler import PathSampler
from squashbert.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--node-cache", required=True, type=Path)
    ap.add_argument("--edge-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200_000)
    args = ap.parse_args()

    print("Loading nodes and caches...")
    nodes = load_nodes(args.nodes)
    cat_index = build_category_index(nodes)
    node_cache = EmbeddingCache.load(args.node_cache)
    edge_cache = EmbeddingCache.load(args.edge_cache)
    with open(args.edge_cache / "edge_types.pkl", "rb") as f:
        edge_types = pickle.load(f)
    sampler = PathSampler(nodes, cat_index, edge_types)
    print(f"  {len(nodes):,} nodes, {len(edge_types):,} edge types")

    embedder = Embedder()
    print(f"Training {args.hops}-hop MLP on {embedder.device}...")

    cfg = TrainConfig(
        n_hops=args.hops,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_size=args.eval_size,
        patience_evals=args.patience,
        max_steps=args.max_steps,
    )
    result = train(cfg, sampler, node_cache, edge_cache, embedder, out_dir=args.out)
    (args.out / "history.json").write_text(json.dumps(result, indent=2))
    print(f"Done. Best held-out cos: {result['best_cos']:.4f}")


if __name__ == "__main__":
    main()
