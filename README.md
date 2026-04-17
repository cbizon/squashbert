# squashbert

Learn small MLPs that map a list of SAPBERT embeddings `[n0, e1, n1, e2, ..., n_n]`
to the sentence-transformer SAPBERT embedding of the corresponding n-hop path sentence.

One MLP per hop count (1, 2, 3).

## Quick start

```bash
uv sync
uv run pytest
```

See `notes/setup.md` for running the full pipeline (including GPU steps) on another machine.

## Rules of the road

See `CLAUDE.md`.
