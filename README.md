# squashbert

Learn small MLPs that map a list of SAPBERT embeddings `[n0, e1, n1, e2, ..., n_n]`
to the sentence-transformer SAPBERT embedding of the corresponding n-hop path sentence.

One MLP per hop count (1, 2, 3).

## Why

Paths through a knowledge graph are scored for "interestingness" by an LLM.
That's too slow and expensive in production, so a downstream proxy model
(RF / MLP) predicts the LLM score from the SAPBERT sentence embedding of the
path. The proxy is fast — but building the sentence embedding on the fly is
still the bottleneck when there are millions of candidate paths.

Node and edge embeddings, on the other hand, can be precomputed once and
cached. If a small MLP can take the concatenated node/edge embeddings and
reproduce the sentence embedding the proxy expects, the entire scoring pipeline
collapses to cache lookups + one tiny forward pass per path.

```
       Path:  n0 ─(e1)─ n1 ─(e2)─ n2
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼

   Slow path (training target)      Fast path (what squashbert learns)

   "n0 e1 n1 e2 n2"                  [emb(n0), emb(e1), emb(n1),
           │                          emb(e2), emb(n2)]
           ▼                                 │   (cached, O(1) lookup)
   SAPBERT sentence encoder                  ▼
           │                          ┌─────────────┐
           ▼                          │  squashbert │  per-hop MLP
   target embedding ◄─── cosine ───►  │     MLP     │
                                      └─────────────┘
                                             │
                                             ▼
                                      predicted embedding
                                             │
                                             ▼
                                      downstream proxy → score
```

Training target: the SAPBERT **sentence** embedding of the rendered path.
Training input: the concatenated SAPBERT **[CLS]** embeddings of each node
and edge phrase. Loss: `1 - cos(pred, target)` on L2-normalized vectors.

## Quick start

```bash
uv sync
uv run pytest
```

See `notes/setup.md` for running the full pipeline (including GPU steps) on another machine.

## Rules of the road

See `CLAUDE.md`.
