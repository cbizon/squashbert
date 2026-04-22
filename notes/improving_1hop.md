# Improving 1-Hop Performance

Current 1-hop performance: **0.93 cosine similarity**

The model trained for 78,000 steps and plateaued, so this is a fundamental ceiling rather than an optimization issue.

## Root Cause Analysis

The training curve shows very slow, steady improvement over 78k steps, suggesting the architecture has hit a capacity or representation limit. Key observation: **inputs and targets use different pooling strategies**:

- **Inputs**: concatenated `[CLS]`-pooled embeddings (node0, edge, node1)
- **Target**: mean-pooled sentence embedding of "node0_name edge_phrase node1_name"

For 1-hop paths, this creates an artificial representation gap - we're concatenating 3 [CLS] vectors and asking an MLP to predict the mean-pooled version of essentially the same content. The model has to "un-[CLS]" the inputs and re-encode them as mean-pooled.

## Recommended Experiments (in priority order)

### 1. **Fix Pooling Mismatch** (Highest Priority) ⭐

Use mean pooling for both inputs and targets. This eliminates the representation gap.

```bash
# 1. Build mean-pooled caches (one-time cost):
uv run python scripts/build_node_cache.py --pooling mean \
    --nodes data/nodes.jsonl --out caches/nodes_mean
uv run python scripts/build_edge_cache.py --pooling mean \
    --nodes data/nodes.jsonl --edges data/edges.jsonl --out caches/edges_mean

# 2. Train (fast — uses caches, same speed as original):
uv run python scripts/train_mlp_mean_pooling.py --hops 1 \
    --nodes data/nodes.jsonl \
    --node-cache caches/nodes_mean \
    --edge-cache caches/edges_mean \
    --out checkpoints/hop1_mean
```

**Expected improvement**: 0.95-0.97+ cosine. This directly addresses the fundamental mismatch.

### 2. **Increase Model Capacity**

Try a deeper, wider architecture with residual connections (4 layers, 2048 hidden).

```bash
uv run python scripts/train_mlp_deep.py --hops 1 \
    --nodes data/nodes.jsonl \
    --node-cache caches/nodes \
    --edge-cache caches/edges \
    --out checkpoints/hop1_deep
```

**Expected improvement**: 0.94-0.95 cosine. Helps if the MLP is capacity-limited, but won't fix the pooling mismatch.

### 3. **Contrastive Loss**

Add negative pairs to encourage more discriminative embeddings.

```bash
uv run python scripts/train_mlp_contrastive.py --hops 1 \
    --nodes data/nodes.jsonl \
    --node-cache caches/nodes \
    --edge-cache caches/edges \
    --out checkpoints/hop1_contrastive \
    --margin 0.5 --alpha 0.1
```

**Expected improvement**: 0.93-0.95 cosine. Helps with retrieval metrics but may not boost raw cosine much.

### 4. **Analyze Failure Cases**

Understand where the model struggles:

```bash
uv run python scripts/analyze_errors.py \
    --checkpoint checkpoints/hop1/best.pt \
    --nodes data/nodes.jsonl \
    --node-cache caches/nodes \
    --edge-cache caches/edges \
    --n-samples 1000 --show-worst 20
```

Look for patterns:
- Are certain edge types consistently low?
- Do long entity names cause issues?
- Are there tokenization artifacts?

## Other Options to Consider

### Learning Dynamics
- **Learning rate schedule**: Warmup + cosine decay instead of constant 3e-4
- **Larger batch size**: 512 or 1024 (more diverse negatives for contrastive loss)
- **Weight decay tuning**: Current default is AdamW's standard 0.01

### Alternative Architectures
- **Transformer encoder** instead of MLP (attention over concatenated embeddings)
- **Gated combinations** (learn to weight node vs edge contributions)
- **Multi-task learning**: predict both final embedding and intermediate representations

### Data Quality
- Review edge rendering logic in `render.py` - ensure natural phrasing
- Check if qualified predicates (causes, affects with qualifiers) are handled well
- Verify entity name preprocessing is consistent

## Quick Win Summary

**Start with the mean pooling approach** (`train_mlp_mean_pooling.py`). This is the most theoretically motivated fix and should give the biggest boost. If you're still not satisfied after that, try the deep architecture or contrastive loss.

If mean pooling gets you to 0.97+, the remaining gap to 1.0 is likely fundamental noise:
- Entity names may be ambiguous or have multiple valid embeddings
- SAPBERT itself may not perfectly distinguish all biomedical concepts
- Random path sampling introduces some train/eval distribution shift

## Comparison Table

| Approach | Expected Cos | Training Time | Complexity |
|----------|-------------|---------------|------------|
| **Current (CLS/mean mismatch)** | 0.93 | 78k steps | Baseline |
| **Mean pooling (inputs+targets)** | 0.95-0.97 | ~50k steps | Easy |
| **Deep MLP (4 layers, 2048)** | 0.94-0.95 | ~100k steps | Medium |
| **Contrastive loss** | 0.93-0.95 | ~80k steps | Medium |
| **Combination: mean + deep** | 0.96-0.98 | ~100k steps | High |
