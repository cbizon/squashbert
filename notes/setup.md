# Running squashbert on another machine

The heavy lifting (building the node embedding cache, training the MLPs) is intended
to run on a CUDA machine, not the dev laptop. This file is the handoff.

## Machine requirements

- Linux + NVIDIA GPU with CUDA 12+ (any consumer card with ≥8 GB VRAM is fine).
- Python 3.11+.
- `uv` installed (https://docs.astral.sh/uv/).
- ~20 GB free disk for caches (2M-node KG: node cache ~3 GB fp16, plus edge-type and
  target-path caches).

## Setup

```bash
git clone https://github.com/cbizon/squashbert.git
cd squashbert
uv sync
uv run pytest                          # fast: 28 tests, no GPU needed
SQUASHBERT_RUN_SLOW=1 uv run pytest    # also runs the SAPBERT + integration tests
```

## Extracting the KGX bundle

The inputs come as a `translator_kg.tar.zst` bundle containing `nodes.jsonl`,
`edges.jsonl`, and `graph-metadata.json`. Decompress once — this writes ~30 GB.

```bash
cd data
tar --use-compress-program=unzstd -xf /path/to/translator_kg.tar.zst
# produces: nodes.jsonl, edges.jsonl, graph-metadata.json
```

All pipeline scripts take plain jsonl paths.

## Pipeline

1. **Node embedding cache** — [CLS]-pooled SAPBERT on every node name.
   ```bash
   uv run python -m squashbert.scripts.build_node_cache \
       --nodes path/to/nodes.jsonl --out caches/nodes
   ```
2. **Edge-type embedding cache** — [CLS]-pooled SAPBERT on each distinct edge-type
   rendering (forward and reverse).
   ```bash
   uv run python -m squashbert.scripts.build_edge_cache \
       --edges path/to/edges.jsonl --nodes path/to/nodes.jsonl --out caches/edges
   ```
3. **Train** — streaming path generation; cosine loss; stop on plateau.
   ```bash
   uv run python -m squashbert.scripts.train_mlp --hops 1 --out checkpoints/hop1
   uv run python -m squashbert.scripts.train_mlp --hops 2 --out checkpoints/hop2
   uv run python -m squashbert.scripts.train_mlp --hops 3 --out checkpoints/hop3
   ```

## Models used

- Node/edge embedder: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `[CLS]` pooling,
  L2-normalized.
- Full-sentence target embedder: same weights loaded as a sentence-transformer
  (mean pooling + L2 normalization).

## Known gotchas

- `bmt` downloads the biolink model YAML on first use; needs network.
- SAPBERT tokenizer truncates at 25 tokens by default — fine for entity names, but for
  the full path sentence we pass a longer `max_length` (see `embed.py`).
