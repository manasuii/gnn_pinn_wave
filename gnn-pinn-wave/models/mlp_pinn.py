"""
mlp_pinn.py
=====================================================================
MLP-PINN: Multilayer Perceptron Physics-Informed Neural Network
for elastic wave propagation.

Supports:
  - 1D wave equation
  - 2D membrane wave equation

Architecture:
  (x,t) or (x,y,t)
      ↓
  Learnable input scaling
      ↓
  FourierEncoder  [same as GNN-PINN — fair comparison]
      ↓
  D × fully-connected layers (deeper to compensate for no GNN)
      ↓
  u(x,t) scalar displacement

KEY DIFFERENCE from GNN-PINN:
  - No graph structure
  - No message passing
  - edge_index is accepted but silently ignored
  - All spatial points treated independently
  - Deeper MLP (8 layers vs 3) for comparable parameter count

This is the baseline used for comparison in:
  "When Does Graph Structure Help in Physics-Informed
   Neural Networks?" — [Your Name], 2025
"""

import torch
import torch.nn as nn
import numpy as np
from layers import FourierEncoder


# ============================================================
# MLP-PINN — 1D Wave Equation
# ============================================================
class MLPPINN1D(nn.Module):
    """
    MLP-PINN for the 1D elastic wave equation:
        ∂²u/∂t² = c(x)² · ∂²u/∂x²

    Inputs: (x, t) per collocation point
    Output: u(x,t) scalar displacement

    No graph structure. All spatial points treated independently.
    Uses the same FourierEncoder as GNNPINN1D for fair comparison.
    Deeper MLP (depth=8) to roughly match GNN-PINN parameter count.

    Args:
        hidden    : hidden dimension
        depth     : number of MLP layers (default 8)
        num_freqs : number of Fourier frequency components

    Example:
        >>> model = MLPPINN1D(hidden=128, depth=8)
        >>> u = model(x, t)         # [N*T]
        >>> u = model(x, t, None)   # also valid (edge_index ignored)
    """
    def __init__(self,
                 hidden:    int = 128,
                 depth:     int = 8,
                 num_freqs: int = 24):
        super().__init__()

        # Learnable per-coordinate scaling
        self.input_scale = nn.Parameter(torch.ones(2))

        # Same Fourier encoder as GNN-PINN — ensures fair comparison
        self.encoder = FourierEncoder(
            input_dim = 2,
            num_freqs = num_freqs,
            hidden    = hidden,
        )

        # Deep MLP — extra layers compensate for no GNN message passing
        # First layer already inside FourierEncoder,
        # so remaining depth = depth - 3 (encoder has 3 layers)
        remaining = max(depth - 3, 1)
        layers = []
        for _ in range(remaining):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                edge_index=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : [N*T]  x-coordinates (float32)
            t          : [N*T]  time values   (float32)
            edge_index : ignored (accepted for API compatibility
                         with GNN-PINN training loops)
        Returns:
            u          : [N*T]  predicted displacement
        """
        # Stack and scale
        xt = torch.stack([x, t], dim=1)        # [N*T, 2]
        xt = xt * self.input_scale

        # Encode with Fourier features
        h = self.encoder(xt)                   # [N*T, hidden]

        # MLP layers
        return self.net(h).squeeze(-1)         # [N*T]


# ============================================================
# MLP-PINN — 2D Membrane Wave Equation
# ============================================================
class MLPPINN2D(nn.Module):
    """
    MLP-PINN for the 2D membrane wave equation:
        ∂²u/∂t² = c² · (∂²u/∂x² + ∂²u/∂y²)

    Inputs: (x, y, t) per node
    Output: u(x,y,t) scalar displacement

    No graph structure. Works on any node layout including
    irregular triangular meshes — but cannot exploit
    mesh topology (unlike GNN-PINN).

    Args:
        hidden    : hidden dimension
        depth     : number of MLP layers
        num_freqs : Fourier frequency components

    Example:
        >>> model = MLPPINN2D(hidden=64, depth=6)
        >>> u = model(x, y, t)            # [N]
        >>> u = model(x, y, t, None)      # also valid
    """
    def __init__(self,
                 hidden:    int = 64,
                 depth:     int = 6,
                 num_freqs: int = 16):
        super().__init__()

        self.input_scale = nn.Parameter(torch.ones(3))

        # 2D encoder: (x,y,t) → hidden
        self.encoder = FourierEncoder(
            input_dim = 3,
            num_freqs = num_freqs,
            hidden    = hidden,
        )

        remaining = max(depth - 3, 1)
        layers = []
        for _ in range(remaining):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                t: torch.Tensor,
                edge_index=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : [N]  x-coordinates
            y          : [N]  y-coordinates
            t          : [N]  time values
            edge_index : ignored
        Returns:
            u          : [N]  predicted displacement
        """
        xyt = torch.stack([x, y, t], dim=1)   # [N, 3]
        xyt = xyt * self.input_scale
        h   = self.encoder(xyt)                # [N, hidden]
        return self.net(h).squeeze(-1)         # [N]


# ============================================================
# PARAMETER COUNT UTILITY
# ============================================================
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    from layers import build_chain_edges, tile_edges

    N, T = 40, 60
    x = torch.rand(N * T)
    t = torch.rand(N * T)
    edge = tile_edges(build_chain_edges(N), N, T)

    m1d = MLPPINN1D()
    u   = m1d(x, t)
    print(f"MLPPINN1D output: {u.shape}  "
          f"params: {count_parameters(m1d):,}")

    x2 = torch.rand(N)
    y2 = torch.rand(N)
    t2 = torch.rand(N)
    m2d = MLPPINN2D()
    u2  = m2d(x2, y2, t2)
    print(f"MLPPINN2D output: {u2.shape}  "
          f"params: {count_parameters(m2d):,}")