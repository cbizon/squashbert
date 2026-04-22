"""Train with a deeper/wider MLP architecture.

Tests whether model capacity is the bottleneck by using:
- 4 hidden layers instead of 2
- 2048 hidden dims instead of 1536
- Residual connections for better gradient flow

Usage:
    uv run python scripts/train_mlp_deep.py --hops 1 \\
        --nodes path/to/nodes.jsonl \\
        --node-cache caches/nodes \\
        --edge-cache caches/edges \\
        --out checkpoints/hop1_deep
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from torch import Tensor, nn

from squashbert.cache import EmbeddingCache
from squashbert.embed import EMBED_DIM, Embedder
from squashbert.kgx import build_category_index, load_nodes
from squashbert.sampler import PathSampler
from squashbert.train import TrainConfig, train


class DeepSquashMLP(nn.Module):
    """Deeper, wider MLP with residual connections.

    4 hidden layers with 2048 dims each. Residual connections help with training depth.
    """

    def __init__(self, n_hops: int, embed_dim: int = 768, hidden: int = 2048):
        super().__init__()
        in_dim = (2 * n_hops + 1) * embed_dim
        self.n_hops = n_hops
        self.embed_dim = embed_dim

        # Project to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # 4 residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            for _ in range(4)
        ])

        # Project to output
        self.output_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)

        # Residual blocks
        for block in self.blocks:
            h = h + block(h)

        y = self.output_proj(h)
        return nn.functional.normalize(y, p=2, dim=1)


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
    print(f"Training {args.hops}-hop DEEP MLP (4 layers, 2048 hidden) on {embedder.device}...")

    # Temporarily swap in the deep model
    from squashbert import model as model_module
    original_model_class = model_module.SquashMLP
    model_module.SquashMLP = DeepSquashMLP

    try:
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
        result["architecture"] = "deep_residual"
        (args.out / "history.json").write_text(json.dumps(result, indent=2))
        print(f"Done. Best held-out cos: {result['best_cos']:.4f}")
    finally:
        model_module.SquashMLP = original_model_class


if __name__ == "__main__":
    main()
