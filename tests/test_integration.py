"""End-to-end smoke test: synthetic KGX -> caches -> train -> eval.

Runs the real SAPBERT model and a few gradient steps. Slow (~1 min on CPU).
Gated behind SQUASHBERT_RUN_SLOW=1 so regular `pytest` stays fast.

The point is to catch wiring bugs (tensor shapes, device/dtype mismatches, cache
key mismatches) that unit tests miss. We do NOT assert any model-quality target —
a handful of gradient steps on a 5-node graph proves nothing about learning.
"""

from __future__ import annotations

import os
import pickle
import random

import numpy as np
import orjson
import pytest

slow = pytest.mark.skipif(
    os.environ.get("SQUASHBERT_RUN_SLOW") != "1",
    reason="set SQUASHBERT_RUN_SLOW=1 to enable",
)


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


@slow
def test_full_pipeline(tmp_path):
    from squashbert.cache import EmbeddingCache, edge_type_key
    from squashbert.embed import EMBED_DIM, Embedder
    from squashbert.eval import evaluate
    from squashbert.kgx import build_category_index, collect_edge_types, load_nodes
    from squashbert.render import render
    from squashbert.sampler import PathSampler
    from squashbert.train import TrainConfig, train

    nodes_p = tmp_path / "nodes.jsonl"
    edges_p = tmp_path / "edges.jsonl"
    _write_jsonl(
        nodes_p,
        [
            {"id": "A:1", "name": "metformin", "category": ["biolink:SmallMolecule"]},
            {"id": "A:2", "name": "aspirin", "category": ["biolink:SmallMolecule"]},
            {"id": "D:1", "name": "diabetes", "category": ["biolink:Disease"]},
            {"id": "D:2", "name": "hypertension", "category": ["biolink:Disease"]},
            {"id": "G:1", "name": "glp1r", "category": ["biolink:Gene"]},
            {"id": "G:2", "name": "insr", "category": ["biolink:Gene"]},
        ],
    )
    _write_jsonl(
        edges_p,
        [
            {"subject": "A:1", "predicate": "biolink:treats", "object": "D:1"},
            {"subject": "A:2", "predicate": "biolink:treats", "object": "D:2"},
            {"subject": "D:1", "predicate": "biolink:related_to", "object": "G:1"},
            {
                "subject": "A:1",
                "predicate": "biolink:affects",
                "qualified_predicate": "biolink:causes",
                "object_aspect_qualifier": "expression",
                "object_direction_qualifier": "decreased",
                "object": "G:2",
            },
        ],
    )

    # Build node + edge caches (mirrors the CLI scripts).
    nodes = load_nodes(nodes_p)
    cat_index = build_category_index(nodes)
    edge_types, skips = collect_edge_types(edges_p, nodes)
    assert skips == {"missing_endpoint": 0, "unsupported_qualifier": 0}

    embedder = Embedder(device="cpu")

    node_ids = list(nodes)
    node_cache = EmbeddingCache.create(
        tmp_path / "ncache", keys=node_ids, dim=EMBED_DIM, dtype="float16"
    )
    names = [nodes[nid].name for nid in node_ids]
    node_cache.vectors[:] = embedder.embed_cls(names).astype(np.float16)
    node_cache.flush()

    edge_keys, edge_phrases = [], []
    for et in edge_types:
        for rev in (False, True):
            edge_keys.append(edge_type_key(et.subject_category, et.spec, et.object_category, rev))
            edge_phrases.append(render(et.spec, reverse=rev))
    edge_cache = EmbeddingCache.create(
        tmp_path / "ecache", keys=edge_keys, dim=EMBED_DIM, dtype="float16"
    )
    edge_cache.vectors[:] = embedder.embed_cls(edge_phrases).astype(np.float16)
    edge_cache.flush()
    with open(tmp_path / "ecache" / "edge_types.pkl", "wb") as f:
        pickle.dump(edge_types, f)

    # Train for a handful of steps with generous plateau settings (we'll hit max_steps).
    sampler = PathSampler(nodes, cat_index, edge_types)
    cfg = TrainConfig(
        n_hops=2,
        batch_size=4,
        eval_every=5,
        eval_size=8,
        patience_evals=100,
        max_steps=15,
    )
    result = train(cfg, sampler, node_cache, edge_cache, embedder, out_dir=tmp_path / "ckpt",
                   device="cpu")
    assert -1.0 <= result["best_cos"] <= 1.0
    assert (tmp_path / "ckpt" / "best.pt").exists()

    # Evaluate from checkpoint.
    from squashbert.eval import load_model

    model = load_model(tmp_path / "ckpt" / "best.pt", device="cpu")
    metrics = evaluate(
        model,
        sampler,
        node_cache,
        edge_cache,
        embedder,
        n_paths=16,
        batch_size=8,
        device="cpu",
    )
    assert metrics["n_paths"] == 16
    assert -1.0 <= metrics["mean_cos"] <= 1.0
    assert 0.0 <= metrics["rank1_in_batch"] <= 1.0
