"""Models that map concatenated [n0, e1, n1, ...] embeddings to the full-path
sentence embedding. One instance per hop count.

All models share the same interface:
- __init__(n_hops, embed_dim=768, ...)
- forward(x: (B, (2*n_hops+1)*D)) -> (B, D), L2-normalized
- .n_hops and .embed_dim attributes
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SquashMLP(nn.Module):
    """Two hidden layers with GELU + LayerNorm. The baseline."""

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


class DeepSquashMLP(nn.Module):
    """Wider with residual blocks for more capacity."""

    def __init__(self, n_hops: int, embed_dim: int = 768, hidden: int = 2048, n_blocks: int = 4):
        super().__init__()
        in_dim = (2 * n_hops + 1) * embed_dim
        self.n_hops = n_hops
        self.embed_dim = embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        y = self.output_proj(h)
        return nn.functional.normalize(y, p=2, dim=1)


class CrossAttentionSquash(nn.Module):
    """Reshape the flat input back to (n_components, D), run cross-attention
    so components can interact, then project to output.

    For 1-hop this is 3 tokens (node, edge, node) — the 3x3 attention is
    essentially free but lets the model learn cross-component interactions
    that a flat MLP has to approximate through nonlinearities.
    """

    def __init__(
        self, n_hops: int, embed_dim: int = 768, n_heads: int = 8, n_layers: int = 2,
    ):
        super().__init__()
        self.n_hops = n_hops
        self.embed_dim = embed_dim
        self.n_components = 2 * n_hops + 1

        # Learned position embeddings for each slot (node0, edge0, node1, ...).
        self.pos_embed = nn.Parameter(torch.randn(self.n_components, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # (B, n_components * D) -> (B, n_components, D)
        h = x.view(B, self.n_components, self.embed_dim)
        h = h + self.pos_embed
        h = self.encoder(h)
        # Mean pool over components -> (B, D)
        y = self.output_proj(h.mean(dim=1))
        return nn.functional.normalize(y, p=2, dim=1)


class CrossAttentionMLPSquash(nn.Module):
    """Cross-attention over components, then flatten and run through an MLP head.

    Same transformer encoder as CrossAttentionSquash, but instead of mean-pooling
    the attended tokens, we flatten them back to (B, n_components * D) and let an
    MLP head map to the output. This preserves per-slot information through to the
    output projection.
    """

    def __init__(
        self,
        n_hops: int,
        embed_dim: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        hidden: int = 1536,
    ):
        super().__init__()
        self.n_hops = n_hops
        self.embed_dim = embed_dim
        self.n_components = 2 * n_hops + 1

        self.pos_embed = nn.Parameter(torch.randn(self.n_components, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        flat_dim = self.n_components * embed_dim
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        h = x.view(B, self.n_components, self.embed_dim)
        h = h + self.pos_embed
        h = self.encoder(h)
        # Flatten attended tokens -> MLP head
        y = self.head(h.reshape(B, -1))
        return nn.functional.normalize(y, p=2, dim=1)


MODELS = {
    "mlp": SquashMLP,
    "deep": DeepSquashMLP,
    "crossattn": CrossAttentionSquash,
    "crossmlp": CrossAttentionMLPSquash,
}


def cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 - cos(pred, target). Both expected already L2-normalized."""
    return (1.0 - (pred * target).sum(dim=1)).mean()
