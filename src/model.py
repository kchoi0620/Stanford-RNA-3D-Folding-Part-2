"""
src/model.py
RNA 3D structure prediction model — Stanford RNA 3D Folding Part 2.

Architecture: Graph Transformer on RNA nucleotide graphs.
  - Each nucleotide is a node with one-hot + positional features
  - Backbone bonds + spatial contacts are edges
  - Multiple Graph Transformer layers aggregate structural context
  - MLP head predicts 3D (x, y, z) per nucleotide

Why graph-based?
  - RNA secondary/tertiary structure is inherently a graph (base pairs, stacking)
  - Equivariant message passing respects 3D geometry
  - torch_geometric provides efficient sparse graph ops on GPU
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool,
    LayerNorm,
)
from torch_geometric.data import Data as PyGData


class RNAGraphTransformer(nn.Module):
    """
    Graph Transformer for RNA 3D coordinate prediction.

    Input:  PyG graph with node features x ∈ R^(L × node_in_dim)
    Output: 3D coordinates per nucleotide ∈ R^(L × 3)

    Args:
        node_in_dim:  Dimension of input node features (default 20: one-hot + pos enc).
        hidden_dim:   Hidden dimension of transformer layers.
        num_layers:   Number of Graph Transformer layers.
        num_heads:    Attention heads in each transformer layer.
        dropout:      Dropout probability.
        edge_dim:     Edge feature dimension (1 for backbone/spatial type).
    """

    def __init__(
        self,
        node_in_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Stack of Graph Transformer layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,       # output: num_heads × (hidden_dim // num_heads) = hidden_dim
                beta=True,         # gating between self and neighborhood
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Feed-forward residual block after each transformer layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # → (x, y, z)
        )

        # Optional: global context injection (graph-level → nodes)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers; zero-init final coord layer bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the final prediction layer for stable training start
        nn.init.zeros_(self.coord_head[-1].weight)
        nn.init.zeros_(self.coord_head[-1].bias)

    def forward(self, data: PyGData) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG batch or single graph with fields:
                  x            (N_total, node_in_dim)
                  edge_index   (2, E)
                  edge_attr    (E, edge_dim)
                  batch        (N_total,)  — provided by DataLoader

        Returns:
            coords: (N_total, 3) predicted 3D coordinates per nucleotide.
        """
        x = self.node_encoder(data.x)           # (N, hidden_dim)
        edge_index = data.edge_index
        edge_attr  = data.edge_attr if data.edge_attr is not None else None

        # Graph Transformer layers with residual connections
        for layer, norm, ffn, ffn_norm in zip(
            self.layers, self.norms, self.ffns, self.ffn_norms
        ):
            # Self-attention message passing
            h = layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x + self.dropout(h))

            # Feed-forward residual
            x = ffn_norm(x + self.dropout(ffn(x)))

        # Inject global (graph-level) mean context into each node
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_ctx = global_mean_pool(x, batch)       # (B, hidden)
        global_ctx = self.global_proj(global_ctx)     # (B, hidden)
        x = x + global_ctx[batch]                     # broadcast to nodes

        # Predict coordinates
        coords = self.coord_head(x)                   # (N, 3)
        return coords


class CoordLoss(nn.Module):
    """
    Combined RMSD-inspired training loss:
      L = α · MSE(pred, true) + (1-α) · Smooth_L1(pairwise_dist_diff)

    The pairwise distance term preserves local geometry even when
    global orientation is not yet aligned.
    """

    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self, pred: torch.Tensor, true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, 3) predicted coordinates.
            true: (N, 3) ground-truth coordinates.
        """
        # Align pred to true via mean centering (differentiable approximation)
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        true_c = true - true.mean(dim=0, keepdim=True)

        coord_loss = self.mse(pred_c, true_c)

        # Pairwise distance preservation (sample 256 pairs for efficiency)
        N = pred.size(0)
        if N > 2:
            n_pairs = min(256, N * (N - 1) // 2)
            idx = torch.randint(0, N, (n_pairs, 2), device=pred.device)
            pred_dists = (pred[idx[:, 0]] - pred[idx[:, 1]]).norm(dim=-1)
            true_dists = (true[idx[:, 0]] - true[idx[:, 1]]).norm(dim=-1)
            dist_loss = self.smooth_l1(pred_dists, true_dists)
        else:
            dist_loss = torch.tensor(0.0, device=pred.device)

        return self.alpha * coord_loss + (1 - self.alpha) * dist_loss
