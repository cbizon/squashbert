"""Analyze where the 1-hop model performs worst.

Samples paths, computes per-example cosine similarity, and reports the lowest-scoring
examples to identify systematic issues.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from squashbert.cache import EmbeddingCache
from squashbert.embed import Embedder
from squashbert.eval import load_model
from squashbert.kgx import build_category_index, load_nodes
from squashbert.sampler import PathSampler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path, help="Path to best.pt")
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--node-cache", required=True, type=Path)
    ap.add_argument("--edge-cache", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=1000, help="Paths to analyze")
    ap.add_argument("--show-worst", type=int, default=20, help="How many worst examples to show")
    args = ap.parse_args()

    print("Loading model and caches...")
    model = load_model(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    nodes = load_nodes(args.nodes)
    cat_index = build_category_index(nodes)
    node_cache = EmbeddingCache.load(args.node_cache)
    edge_cache = EmbeddingCache.load(args.edge_cache)
    with open(args.edge_cache / "edge_types.pkl", "rb") as f:
        edge_types = pickle.load(f)
    sampler = PathSampler(nodes, cat_index, edge_types)
    embedder = Embedder(device=model.net[0].weight.device.type)

    print(f"Sampling {args.n_samples} paths...")
    rng = random.Random(42)
    paths = [sampler.sample(model.n_hops, rng) for _ in range(args.n_samples)]

    # Stack inputs and targets
    def stack_inputs(paths_list):
        if not paths_list:
            return np.empty((0, 0), dtype=np.float32)
        n_hops = len(paths_list[0].edge_phrases)
        dim = node_cache.dim
        out = np.empty((len(paths_list), (2 * n_hops + 1) * dim), dtype=np.float32)
        for i, p in enumerate(paths_list):
            row = out[i]
            pos = 0
            for h in range(n_hops):
                row[pos : pos + dim] = node_cache[p.node_ids[h]]
                pos += dim
                row[pos : pos + dim] = edge_cache[p.edge_keys[h]]
                pos += dim
            row[pos : pos + dim] = node_cache[p.node_ids[n_hops]]
        return out

    x = torch.from_numpy(stack_inputs(paths)).to(embedder.device)
    sentences = [p.sentence for p in paths]
    y = torch.from_numpy(embedder.embed_sentence(sentences, batch_size=128)).to(embedder.device)

    print("Computing predictions...")
    model.eval()
    with torch.inference_mode():
        pred = model(x)

    # Per-example cosine similarity
    cos_sim = (pred * y).sum(dim=1).cpu().numpy()

    print(f"\nOverall statistics:")
    print(f"  Mean cosine: {cos_sim.mean():.4f}")
    print(f"  Std cosine:  {cos_sim.std():.4f}")
    print(f"  Min cosine:  {cos_sim.min():.4f}")
    print(f"  Max cosine:  {cos_sim.max():.4f}")
    print(f"  10th percentile: {np.percentile(cos_sim, 10):.4f}")
    print(f"  90th percentile: {np.percentile(cos_sim, 90):.4f}")

    # Show worst examples
    worst_indices = np.argsort(cos_sim)[: args.show_worst]
    print(f"\n{args.show_worst} worst examples:")
    for rank, idx in enumerate(worst_indices, 1):
        p = paths[idx]
        print(f"\n#{rank} - Cosine: {cos_sim[idx]:.4f}")
        print(f"  Path: {p.sentence}")
        print(f"  Node IDs: {' -> '.join(p.node_ids)}")
        print(f"  Edge phrases: {p.edge_phrases}")


if __name__ == "__main__":
    main()
