"""
layers.py
=====================================================================
Shared building blocks used by both GNN-PINN and MLP-PINN.

Contents:
  - FourierEncoder    : random Fourier feature encoder (1D and 2D)
  - WaveGNNLayer      : mean-aggregation message passing layer
  - MaxAggLayer       : max-aggregation message passing layer
  - SumAggLayer       : sum-aggregation message passing layer
  - AttentionAggLayer : GAT-based attention aggregation layer
  - build_chain_edges : builds 1D chain graph edge_index
  - tile_edges        : tiles edge_index across time steps
  - make_c_smooth     : smooth sigmoidal wave speed field
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing, GATConv


# ============================================================
# FOURIER FEATURE ENCODER
# ============================================================
class FourierEncoder(nn.Module):
    """
    Maps input coordinates to random Fourier features.

    For 1D wave:  input_dim=2  (x, t)
    For 2D wave:  input_dim=3  (x, y, t)

    The random matrix B is fixed after initialisation (not trained).
    This directly addresses spectral bias in oscillatory solutions.

    Reference:
        Tancik et al. (2020). Fourier Features Let Networks Learn
        High Frequency Functions in Low Dimensional Domains. NeurIPS.

    Args:
        input_dim  : number of input coordinates (2 for 1D, 3 for 2D)
        num_freqs  : number of frequency components
        hidden     : output embedding dimension
        scale      : std of frequency sampling (default π)
    """
    def __init__(self, input_dim: int = 2,
                 num_freqs: int = 24,
                 hidden: int = 128,
                 scale: float = np.pi):
        super().__init__()
        B = torch.randn(input_dim, num_freqs) * scale
        self.register_buffer('B', B)               # fixed, not trained
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(num_freqs * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),        nn.Tanh(),
            nn.Linear(hidden, hidden),        nn.Tanh(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords : [N, input_dim]  float32 coordinate tensor
        Returns:
            h      : [N, hidden]     encoded features
        """
        proj = coords @ self.B                          # [N, num_freqs]
        feat = torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)     # [N, 2*num_freqs]
        return self.net(feat)                           # [N, hidden]


# ============================================================
# GNN MESSAGE PASSING LAYERS
# ============================================================
class WaveGNNLayer(MessagePassing):
    """
    Standard mean-aggregation message passing layer.

    Each node aggregates information from its spatial neighbours:
        h_i ← h_i + MLP([h_i, MEAN_{j∈N(i)} MLP_msg([h_i, h_j])])

    Used in the default GNN-PINN architecture.

    Args:
        hidden : hidden dimension (must match encoder output)
    """
    def __init__(self, hidden: int = 128):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor,
                x_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class MaxAggLayer(MessagePassing):
    """
    Max-aggregation message passing layer.

    Uses element-wise maximum instead of mean. Less smoothing,
    captures extremes. Can be more expressive than mean for
    detecting sharp features.

    Args:
        hidden : hidden dimension
    """
    def __init__(self, hidden: int = 128):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor,
                x_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class SumAggLayer(MessagePassing):
    """
    Sum-aggregation message passing layer.

    Degree-sensitive: nodes with more neighbours receive larger
    aggregated signals. More expressive than mean for graph
    isomorphism tasks (Xu et al., 2019).

    Args:
        hidden : hidden dimension
    """
    def __init__(self, hidden: int = 128):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor,
                x_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class AttentionAggLayer(nn.Module):
    """
    Graph Attention (GAT) aggregation layer.

    Learns per-neighbour attention weights α_ij, allowing the
    model to focus on the most relevant spatial neighbours.
    Most expressive aggregation strategy tested.

    Reference:
        Veličković et al. (2018). Graph Attention Networks. ICLR.

    Args:
        hidden : hidden dimension
        heads  : number of attention heads (output dim = hidden)
    """
    def __init__(self, hidden: int = 128, heads: int = 4):
        super().__init__()
        assert hidden % heads == 0, \
            "hidden must be divisible by heads"
        self.gat = GATConv(
            in_channels  = hidden,
            out_channels = hidden // heads,
            heads        = heads,
            concat       = True,    # output dim = heads * (hidden//heads) = hidden
            dropout      = 0.0,
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.act(self.gat(x, edge_index))


# ============================================================
# GRAPH CONSTRUCTION UTILITIES
# ============================================================
def build_chain_edges(N: int) -> torch.Tensor:
    """
    Build undirected chain graph edge_index for N nodes.

    Nodes: 0 — 1 — 2 — ... — N-1
    Each adjacent pair connected in both directions.

    Args:
        N : number of spatial nodes
    Returns:
        edge_index : [2, 2*(N-1)]  long tensor
    """
    src, dst = [], []
    for i in range(N - 1):
        src += [i,   i+1]
        dst += [i+1, i  ]
    return torch.tensor([src, dst], dtype=torch.long)


def tile_edges(edge_index: torch.Tensor,
               N: int, T: int) -> torch.Tensor:
    """
    Replicate a single-slice edge_index across T time steps.

    For a time-space collocation grid of shape [T*N], the
    edge_index for time slice t is offset by t*N.
    Time slices remain independent (no temporal edges).

    Args:
        edge_index : [2, E]  edge_index for one time slice
        N          : number of spatial nodes
        T          : number of time steps
    Returns:
        edge_index_full : [2, T*E]  tiled edge_index
    """
    return torch.cat(
        [edge_index + t * N for t in range(T)], dim=1)


def make_c_smooth(x_nodes: torch.Tensor,
                  c1: float = 1.0,
                  c2: float = 1.5,
                  center: float = 0.5,
                  width: float = 0.03) -> torch.Tensor:
    """
    Smooth sigmoidal wave speed profile for bimetallic rod.

    c(x) = c1 + (c2-c1) * sigmoid((x - center) / width)

    Physical interpretation:
        c1 = wave speed in material 1 (e.g. steel, normalised)
        c2 = wave speed in material 2 (e.g. aluminium, normalised)
        center = interface location
        width  = transition zone width (smooth bonded joint)

    The smooth profile ensures c(x) is differentiable everywhere,
    enabling correct automatic differentiation through the PDE loss.

    Args:
        x_nodes : [N]  spatial node positions
        c1      : wave speed left of interface
        c2      : wave speed right of interface
        center  : interface centre location
        width   : sigmoid transition width
    Returns:
        c_field : [N]  wave speed at each node
    """
    return c1 + (c2 - c1) * torch.sigmoid(
        (x_nodes - center) / width)