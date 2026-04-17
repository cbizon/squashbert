"""Streaming training loop for one hop-count MLP.

On each step:

1. Sample `batch_size` random n-hop paths from the KG schema.
2. Look up [CLS]-pooled cached embeddings for each node and edge phrase, concatenate.
3. Embed the full-path sentence with SAPBERT mean-pooling → target.
4. Forward MLP, cosine loss, backprop.

Plateau detection: every `eval_every` steps, sample `eval_size` held-out paths with a
fixed RNG and measure mean cosine. If the best value hasn't improved in
`patience_evals` consecutive evaluations, stop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .cache import EmbeddingCache
from .embed import EMBED_DIM, Embedder
from .model import SquashMLP, cosine_loss
from .sampler import PathSampler


@dataclass
class TrainConfig:
    n_hops: int
    batch_size: int = 256
    lr: float = 3e-4
    eval_every: int = 500
    eval_size: int = 1024
    patience_evals: int = 5
    max_steps: int = 200_000
    eval_rng_seed: int = 123
    train_rng_seed: int = 0


def _stack_inputs(
    paths: list, node_cache: EmbeddingCache, edge_cache: EmbeddingCache
) -> np.ndarray:
    """(B, (2n+1) * D). Alternating node, edge, node, edge, ..., node."""
    if not paths:
        return np.empty((0, 0), dtype=np.float32)
    n_hops = len(paths[0].edge_phrases)
    dim = node_cache.dim
    out = np.empty((len(paths), (2 * n_hops + 1) * dim), dtype=np.float32)
    for i, p in enumerate(paths):
        row = out[i]
        pos = 0
        for h in range(n_hops):
            row[pos : pos + dim] = node_cache[p.node_ids[h]]
            pos += dim
            row[pos : pos + dim] = edge_cache[p.edge_keys[h]]
            pos += dim
        row[pos : pos + dim] = node_cache[p.node_ids[n_hops]]
    return out


def _embed_targets(paths: list, embedder: Embedder, batch_size: int) -> np.ndarray:
    sentences = [p.sentence for p in paths]
    return embedder.embed_sentence(sentences, batch_size=batch_size)


def _eval_mean_cosine(
    model: SquashMLP,
    sampler: PathSampler,
    node_cache: EmbeddingCache,
    edge_cache: EmbeddingCache,
    embedder: Embedder,
    cfg: TrainConfig,
    device: str,
) -> float:
    model.eval()
    rng = random.Random(cfg.eval_rng_seed)
    paths = [sampler.sample(cfg.n_hops, rng) for _ in range(cfg.eval_size)]
    x = torch.from_numpy(_stack_inputs(paths, node_cache, edge_cache)).to(device)
    y = torch.from_numpy(_embed_targets(paths, embedder, cfg.batch_size)).to(device)
    with torch.inference_mode():
        pred = model(x)
    cos = (pred * y).sum(dim=1).mean().item()
    model.train()
    return cos


def train(
    cfg: TrainConfig,
    sampler: PathSampler,
    node_cache: EmbeddingCache,
    edge_cache: EmbeddingCache,
    embedder: Embedder,
    out_dir: str | Path,
    device: str | None = None,
) -> dict:
    """Train until plateau (or max_steps). Saves best checkpoint to `out_dir/best.pt`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or embedder.device

    model = SquashMLP(n_hops=cfg.n_hops, embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()

    rng = random.Random(cfg.train_rng_seed)
    best_cos = -1.0
    evals_since_best = 0
    history: list[dict] = []

    pbar = tqdm(range(cfg.max_steps), desc=f"{cfg.n_hops}-hop")
    for step in pbar:
        paths = [sampler.sample(cfg.n_hops, rng) for _ in range(cfg.batch_size)]
        x = torch.from_numpy(_stack_inputs(paths, node_cache, edge_cache)).to(device)
        y = torch.from_numpy(_embed_targets(paths, embedder, cfg.batch_size)).to(device)

        pred = model(x)
        loss = cosine_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (step + 1) % cfg.eval_every == 0:
            cos = _eval_mean_cosine(
                model, sampler, node_cache, edge_cache, embedder, cfg, device
            )
            history.append({"step": step + 1, "eval_cos": cos})
            if cos > best_cos:
                best_cos = cos
                evals_since_best = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "n_hops": cfg.n_hops,
                        "step": step + 1,
                        "best_cos": best_cos,
                    },
                    out_dir / "best.pt",
                )
            else:
                evals_since_best += 1
            if evals_since_best >= cfg.patience_evals:
                break
    return {"best_cos": best_cos, "history": history}
