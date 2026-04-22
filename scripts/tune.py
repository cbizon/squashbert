"""Optuna hyperparameter search over model architecture and training config.

Searches learning rate, batch size, hidden dim, depth, and architecture.
Uses median pruning to kill unpromising trials early.

Usage:
    uv run python scripts/tune.py --hops 1 \
        --nodes ../translator_kg/April_4/nodes.jsonl \
        --node-cache caches/nodes \
        --edge-cache caches/edges \
        --out checkpoints/tune_hop1 \
        --n-trials 40
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import optuna

from squashbert.cache import EmbeddingCache
from squashbert.embed import Embedder
from squashbert.kgx import build_category_index, load_nodes
from squashbert.sampler import PathSampler
from squashbert.train import TrainConfig, train


def make_objective(
    hops: int,
    sampler: PathSampler,
    node_cache: EmbeddingCache,
    edge_cache: EmbeddingCache,
    embedder: Embedder,
    out_dir: Path,
    max_steps: int,
):
    def objective(trial: optuna.Trial) -> float:
        # --- Search space ---
        model_name = trial.suggest_categorical("model", ["deep", "crossmlp"])
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        hidden = trial.suggest_categorical("hidden", [1024, 1536, 2048, 3072])

        model_kwargs: dict = {"hidden": hidden}

        if model_name == "deep":
            n_blocks = trial.suggest_int("n_blocks", 2, 6)
            model_kwargs["n_blocks"] = n_blocks
        elif model_name == "crossmlp":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            n_heads = trial.suggest_categorical("n_heads", [4, 8])
            model_kwargs["n_layers"] = n_layers
            model_kwargs["n_heads"] = n_heads

        trial_dir = out_dir / f"trial_{trial.number:03d}"

        cfg = TrainConfig(
            n_hops=hops,
            batch_size=batch_size,
            lr=lr,
            eval_every=500,
            eval_size=1024,
            patience_evals=5,
            max_steps=max_steps,
        )

        result = train(
            cfg,
            sampler,
            node_cache,
            edge_cache,
            embedder,
            out_dir=trial_dir,
            model_name=model_name,
            model_kwargs=model_kwargs,
            trial=trial,
        )

        # Save trial details
        trial_info = {
            "trial": trial.number,
            "params": trial.params,
            **result,
        }
        (trial_dir / "history.json").write_text(json.dumps(trial_info, indent=2))

        return result["best_cos"]

    return objective


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--node-cache", required=True, type=Path)
    ap.add_argument("--edge-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-trials", type=int, default=40)
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
    print(f"Running {args.n_trials} trials on {embedder.device}...")

    args.out.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"squash_{args.hops}hop",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///{args.out / 'study.db'}",
        load_if_exists=True,
    )

    objective = make_objective(
        args.hops, sampler, node_cache, edge_cache, embedder, args.out, args.max_steps,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Cosine: {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=4)}")

    # Save summary
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial": best.number,
        "n_trials": len(study.trials),
    }
    (args.out / "best.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
