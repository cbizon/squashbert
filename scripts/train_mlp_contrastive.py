"""Train with contrastive loss instead of simple cosine loss.

In addition to making pred similar to its target, we also push pred to be dissimilar
to other targets in the batch. This should encourage the model to produce more
discriminative embeddings.

Loss: (1 - cos(pred_i, target_i)) + α * mean_j≠i(max(0, cos(pred_i, target_j) - margin))

Usage:
    uv run python scripts/train_mlp_contrastive.py --hops 1 \\
        --nodes path/to/nodes.jsonl \\
        --node-cache caches/nodes \\
        --edge-cache caches/edges \\
        --out checkpoints/hop1_contrastive
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from squashbert.cache import EmbeddingCache
from squashbert.embed import EMBED_DIM, Embedder
from squashbert.kgx import build_category_index, load_nodes
from squashbert.model import SquashMLP
from squashbert.sampler import PathSampler
from squashbert.train import TrainConfig, _embed_targets, _stack_inputs


def contrastive_loss(
    pred: Tensor, target: Tensor, margin: float = 0.5, alpha: float = 0.1
) -> Tensor:
    """Contrastive cosine loss.

    Args:
        pred: (B, D) predictions
        target: (B, D) targets
        margin: similarity threshold below which negative pairs don't contribute
        alpha: weight for the contrastive term

    Returns:
        Scalar loss
    """
    # Positive loss: make pred similar to target
    pos_loss = (1.0 - (pred * target).sum(dim=1)).mean()

    # Negative loss: make pred dissimilar to other targets in batch
    # sim_matrix[i, j] = cos(pred_i, target_j)
    sim_matrix = pred @ target.T  # (B, B)

    # Mask out diagonal (positive pairs)
    batch_size = pred.shape[0]
    mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=pred.device)

    # Hinge loss: penalize similarities above margin
    neg_similarities = sim_matrix * mask
    neg_loss = torch.clamp(neg_similarities - margin, min=0.0).sum() / (
        batch_size * (batch_size - 1)
    )

    return pos_loss + alpha * neg_loss


def train_contrastive(
    cfg: TrainConfig,
    sampler: PathSampler,
    node_cache: EmbeddingCache,
    edge_cache: EmbeddingCache,
    embedder: Embedder,
    out_dir: Path,
    margin: float = 0.5,
    alpha: float = 0.1,
    device: str | None = None,
) -> dict:
    """Train with contrastive loss."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or embedder.device

    model = SquashMLP(n_hops=cfg.n_hops, embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()

    rng = random.Random(cfg.train_rng_seed)
    eval_rng = random.Random(cfg.eval_rng_seed)
    best_cos = -1.0
    evals_since_best = 0
    history = []

    pbar = tqdm(range(cfg.max_steps), desc=f"{cfg.n_hops}-hop (contrastive)")
    for step in pbar:
        paths = [sampler.sample(cfg.n_hops, rng) for _ in range(cfg.batch_size)]
        x = torch.from_numpy(_stack_inputs(paths, node_cache, edge_cache)).to(device)
        y = torch.from_numpy(_embed_targets(paths, embedder, cfg.batch_size)).to(device)

        pred = model(x)
        loss = contrastive_loss(pred, y, margin=margin, alpha=alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (step + 1) % cfg.eval_every == 0:
            model.eval()
            eval_paths = [sampler.sample(cfg.n_hops, eval_rng) for _ in range(cfg.eval_size)]
            eval_x = torch.from_numpy(
                _stack_inputs(eval_paths, node_cache, edge_cache)
            ).to(device)
            eval_y = torch.from_numpy(
                _embed_targets(eval_paths, embedder, cfg.batch_size)
            ).to(device)
            with torch.inference_mode():
                eval_pred = model(eval_x)
            cos = (eval_pred * eval_y).sum(dim=1).mean().item()
            model.train()

            history.append({"step": step + 1, "eval_cos": cos})
            pbar.set_postfix(loss=f"{loss.item():.4f}", eval_cos=f"{cos:.4f}")

            if cos > best_cos:
                best_cos = cos
                evals_since_best = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "n_hops": cfg.n_hops,
                        "step": step + 1,
                        "best_cos": best_cos,
                        "loss": "contrastive",
                        "margin": margin,
                        "alpha": alpha,
                    },
                    out_dir / "best.pt",
                )
            else:
                evals_since_best += 1

            if evals_since_best >= cfg.patience_evals:
                print(f"\nEarly stopping at step {step + 1}")
                break

    return {
        "best_cos": best_cos,
        "history": history,
        "loss": "contrastive",
        "margin": margin,
        "alpha": alpha,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--node-cache", required=True, type=Path)
    ap.add_argument("--edge-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.5, help="Contrastive margin")
    ap.add_argument("--alpha", type=float, default=0.1, help="Contrastive weight")
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
    print(f"Training {args.hops}-hop MLP with contrastive loss on {embedder.device}...")
    print(f"  margin={args.margin}, alpha={args.alpha}")

    cfg = TrainConfig(
        n_hops=args.hops,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_size=args.eval_size,
        patience_evals=args.patience,
        max_steps=args.max_steps,
    )
    result = train_contrastive(
        cfg,
        sampler,
        node_cache,
        edge_cache,
        embedder,
        out_dir=args.out,
        margin=args.margin,
        alpha=args.alpha,
    )
    (args.out / "history.json").write_text(json.dumps(result, indent=2))
    print(f"Done. Best held-out cos: {result['best_cos']:.4f}")


if __name__ == "__main__":
    main()
