"""SAPBERT embedding in two pooling regimes.

Nodes and edge phrases are short noun/verb phrases — we embed them with `[CLS]`
pooling, matching how SAPBERT was trained. Full-path sentences are longer and
off-distribution for SAPBERT; we embed them with mean pooling (the standard
sentence-transformers recipe), which gives a more balanced whole-string
representation. Both outputs are L2-normalized so cosine similarity is a dot
product.

Same weights, two different reductions — the MLP's job is to bridge the
resulting gap.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

SAPBERT_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
EMBED_DIM = 768


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Load SAPBERT once, embed in either pooling regime.

    Not a fancy abstraction — just carries the tokenizer, model, and device so the
    caller doesn't pass them around. Still plain functions for the pooling ops.
    """

    def __init__(self, device: str | None = None, dtype: torch.dtype = torch.float32):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(SAPBERT_NAME)
        self.model = AutoModel.from_pretrained(SAPBERT_NAME, torch_dtype=dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_cls(
        self, strings: list[str], batch_size: int = 128, max_length: int = 32
    ) -> np.ndarray:
        """`[CLS]`-pooled, L2-normalized embeddings. Use for entities and edge phrases."""
        return self._embed(strings, batch_size, max_length, pool="cls")

    @torch.inference_mode()
    def embed_sentence(
        self, strings: list[str], batch_size: int = 64, max_length: int = 128
    ) -> np.ndarray:
        """Mean-pooled (masked), L2-normalized. Use for full path sentences."""
        return self._embed(strings, batch_size, max_length, pool="mean")

    def _embed(
        self, strings: list[str], batch_size: int, max_length: int, pool: str
    ) -> np.ndarray:
        out = np.empty((len(strings), EMBED_DIM), dtype=np.float32)
        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state  # [B, T, 768]
            if pool == "cls":
                vecs = hidden[:, 0]
            elif pool == "mean":
                vecs = _masked_mean(hidden, enc["attention_mask"])
            else:
                raise ValueError(pool)
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            out[i : i + len(batch)] = vecs.float().cpu().numpy()
        return out


def _masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    """Mean over non-padding tokens."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
