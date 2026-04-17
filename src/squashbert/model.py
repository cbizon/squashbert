"""The MLP that maps concatenated [n0, e1, n1, ...] embeddings to the full-path
sentence embedding. One instance per hop count.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SquashMLP(nn.Module):
    """Input: (B, (2*n_hops + 1) * D). Output: (B, D), L2-normalized.

    Two hidden layers with GELU + LayerNorm. Small enough to train fast, big enough
    to have room to fit the compositional mapping we're after.
    """

    def __init__(self, n_hops: int, embed_dim: int = 768, hidden: int = 1536):
        super().__init__()
        in_dim = (2 * n_hops + 1) * embed_dim
        self.n_hops = n_hops
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return nn.functional.normalize(y, p=2, dim=1)


def cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 - cos(pred, target). Both expected already L2-normalized."""
    return (1.0 - (pred * target).sum(dim=1)).mean()
