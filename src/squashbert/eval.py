"""Evaluate a trained MLP on a held-out sample of paths.

Reports two metrics:
- mean cosine(pred, target): direct fit quality.
- rank-in-batch@1: for each path in a batch, how often the target is the predicted
  vector's nearest neighbor *within that batch* (a retrieval-style sanity check —
  harder than cosine because it punishes near-duplicates).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import EmbeddingCache
from .embed import EMBED_DIM, Embedder
from .model import MODELS, SquashMLP
from .sampler import PathSampler
from .train import _embed_targets, _stack_inputs


def load_model(ckpt_path: str | Path, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "mlp")
    model_cls = MODELS[model_name]
    model = model_cls(n_hops=ckpt["n_hops"], embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate(
    model: SquashMLP,
    sampler: PathSampler,
    node_cache: EmbeddingCache,
    edge_cache: EmbeddingCache,
    embedder: Embedder,
    n_paths: int = 2048,
    batch_size: int = 256,
    seed: int = 999,
    device: str | None = None,
) -> dict:
    device = device or embedder.device
    rng = random.Random(seed)
    paths = [sampler.sample(model.n_hops, rng) for _ in range(n_paths)]

    all_cos = []
    all_rank1 = 0
    total = 0
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        x = torch.from_numpy(_stack_inputs(batch, node_cache, edge_cache)).to(device)
        y = torch.from_numpy(_embed_targets(batch, embedder, batch_size)).to(device)
        with torch.inference_mode():
            pred = model(x)
        cos = (pred * y).sum(dim=1)
        all_cos.append(cos.cpu().numpy())

        # Rank-in-batch@1: argmax of pred @ y.T should be the diagonal.
        sim = pred @ y.T  # (b, b)
        pred_idx = sim.argmax(dim=1)
        correct = (pred_idx == torch.arange(len(batch), device=device)).sum().item()
        all_rank1 += correct
        total += len(batch)

    return {
        "mean_cos": float(np.concatenate(all_cos).mean()),
        "rank1_in_batch": all_rank1 / total,
        "n_paths": total,
    }
