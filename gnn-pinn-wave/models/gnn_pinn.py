"""
gnn_pinn.py
=====================================================================
GNN-PINN: Graph Neural Network Physics-Informed Neural Network
for elastic wave propagation.

Supports:
  - 1D wave equation (chain graph)
  - 2D membrane wave equation (irregular triangular mesh)
  - Multiple aggregation strategies: mean, max, sum, attention

Architecture:
  (x,t) or (x,y,t)
      ↓
  Learnable input scaling
      ↓
  FourierEncoder  [shared with MLP-PINN]
      ↓
  L × GNN message-passing layers (with residual connections)
      ↓
  Decoder MLP
      ↓
  u(x,t) scalar displacement

Reference:
  "When Does Graph Structure Help in Physics-Informed
   Neural Networks?" — [Your Name], 2025
"""

import torch
import torch.nn as nn
from layers import (FourierEncoder, WaveGNNLayer,
                    MaxAggLayer, SumAggLayer, AttentionAggLayer)


# ── Aggregation layer factory ─────────────────────────────────
def _make_layer(agg_type: str, hidden: int,
                heads: int = 4) -> nn.Module:
    """Return the correct aggregation layer for agg_type."""
    if agg_type == 'mean':
        return WaveGNNLayer(hidden)
    elif agg_type == 'max':
        return MaxAggLayer(hidden)
    elif agg_type == 'sum':
        return SumAggLayer(hidden)
    elif agg_type == 'attention':
        return AttentionAggLayer(hidden, heads=heads)
    else:
        raise ValueError(
            f"Unknown agg_type '{agg_type}'. "
            f"Choose from: mean, max, sum, attention")


# ============================================================
# GNN-PINN — 1D Wave Equation
# ============================================================
class GNNPINN1D(nn.Module):
    """
    GNN-PINN for the 1D elastic wave equation:
        ∂²u/∂t² = c(x)² · ∂²u/∂x²

    Inputs per node: (x, t)  — spatial position + time
    Output per node: u(x,t)  — scalar displacement

    The spatial domain is a chain graph with N nodes.
    For training, the graph is tiled across T time steps
    using tile_edges() from layers.py.

    Args:
        hidden     : hidden dimension for all layers
        num_layers : number of GNN message-passing layers
        num_freqs  : number of Fourier frequency components
        agg_type   : aggregation strategy
                     ('mean' | 'max' | 'sum' | 'attention')
        attn_heads : attention heads (only used if attention)

    Example:
        >>> model = GNNPINN1D(hidden=128, num_layers=6)
        >>> u = model(x, t, edge_index)   # [N*T]
    """
    def __init__(self,
                 hidden:     int = 128,
                 num_layers: int = 6,
                 num_freqs:  int = 24,
                 agg_type:   str = 'mean',
                 attn_heads: int = 4):
        super().__init__()

        # Learnable per-coordinate scaling (helps on periodic solutions)
        self.input_scale = nn.Parameter(torch.ones(2))

        # Fourier encoder: (x,t) → hidden embedding
        self.encoder = FourierEncoder(
            input_dim = 2,
            num_freqs = num_freqs,
            hidden    = hidden,
        )

        # GNN message-passing layers with residual connections
        self.gnn_layers = nn.ModuleList([
            _make_layer(agg_type, hidden, attn_heads)
            for _ in range(num_layers)
        ])

        # Decoder: hidden → scalar displacement
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        # Xavier initialisation for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : [N*T]     x-coordinates (float32)
            t          : [N*T]     time values   (float32)
            edge_index : [2, E]    graph edges (tiled across time)
        Returns:
            u          : [N*T]     predicted displacement
        """
        # Stack and scale inputs
        xt = torch.stack([x, t], dim=1)        # [N*T, 2]
        xt = xt * self.input_scale              # learnable scaling

        # Encode to hidden space
        h = self.encoder(xt)                   # [N*T, hidden]

        # GNN message passing with residual connections
        for layer in self.gnn_layers:
            h = h + layer(h, edge_index)       # residual: h ← h + Δh

        # Decode to displacement
        return self.decoder(h).squeeze(-1)     # [N*T]


# ============================================================
# GNN-PINN — 2D Membrane Wave Equation
# ============================================================
class GNNPINN2D(nn.Module):
    """
    GNN-PINN for the 2D membrane wave equation:
        ∂²u/∂t² = c² · (∂²u/∂x² + ∂²u/∂y²)

    Inputs per node: (x, y, t)  — 2D position + time
    Output per node: u(x,y,t)   — scalar displacement

    The spatial domain is an irregular Delaunay triangulation.
    Edges are built using scipy.spatial.Delaunay and passed
    as edge_index. See experiments/2d_membrane/extensions_code.py
    for mesh construction.

    Args:
        hidden     : hidden dimension
        num_layers : number of GNN layers
        num_freqs  : Fourier frequencies
        agg_type   : aggregation ('mean'|'max'|'sum'|'attention')
        attn_heads : attention heads (only if attention)

    Example:
        >>> model = GNNPINN2D(hidden=64, num_layers=4)
        >>> u = model(x, y, t, edge_index)   # [N]
    """
    def __init__(self,
                 hidden:     int = 64,
                 num_layers: int = 4,
                 num_freqs:  int = 16,
                 agg_type:   str = 'mean',
                 attn_heads: int = 4):
        super().__init__()

        self.input_scale = nn.Parameter(torch.ones(3))

        # 2D encoder: (x,y,t) → hidden
        self.encoder = FourierEncoder(
            input_dim = 3,
            num_freqs = num_freqs,
            hidden    = hidden,
        )

        self.gnn_layers = nn.ModuleList([
            _make_layer(agg_type, hidden, attn_heads)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                t: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : [N]    x-coordinates
            y          : [N]    y-coordinates
            t          : [N]    time values
            edge_index : [2, E] graph edges (triangular mesh)
        Returns:
            u          : [N]    predicted displacement
        """
        xyt = torch.stack([x, y, t], dim=1)   # [N, 3]
        xyt = xyt * self.input_scale
        h   = self.encoder(xyt)                # [N, hidden]
        for layer in self.gnn_layers:
            h = h + layer(h, edge_index)
        return self.decoder(h).squeeze(-1)     # [N]