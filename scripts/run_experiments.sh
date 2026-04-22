#!/usr/bin/env bash
set -euo pipefail

# Deep (4 residual blocks, 2048 wide)
#uv run python scripts/train_mlp.py --model deep --hops 1 \
#    --nodes ../translator_kg/April_4/nodes.jsonl \
#    --node-cache caches/nodes \
#    --edge-cache caches/edges \
#    --out checkpoints/hop1_deep

# Cross-attention (2-layer transformer over 3 component tokens)
#uv run python scripts/train_mlp.py --model crossattn --hops 1 \
#    --nodes ../translator_kg/April_4/nodes.jsonl \
#    --node-cache caches/nodes \
#    --edge-cache caches/edges \
#    --out checkpoints/hop1_crossattn

# Cross-attention + flatten + MLP head (no mean pooling)
uv run python scripts/train_mlp.py --model crossmlp --hops 1 \
    --nodes ../translator_kg/April_4/nodes.jsonl \
    --node-cache caches/nodes \
    --edge-cache caches/edges \
    --out checkpoints/hop1_crossmlp
