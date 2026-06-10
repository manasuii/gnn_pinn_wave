"""
GNN-PINN Extensions Package
=====================================================================
Three self-contained extension modules:

  Module 1: 2D Membrane Wave Equation on Irregular Mesh
  Module 2: Training Diagnostics (loss curves + gradient norms)
  Module 3: Alternative GNN Aggregation Strategies

Run each module independently. All outputs saved as .png and .npy
for inclusion in your paper.

REQUIREMENTS:
  pip install torch torch-geometric matplotlib numpy scipy
"""

# ============================================================
# █ Module 1
# 2D MEMBRANE WAVE EQUATION ON IRREGULAR TRIANGULAR MESH
# PDE: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)  on unit square
# Mesh: Delaunay triangulation (irregular — GNN advantage expected)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from torch_geometric.nn import MessagePassing, GATConv
import os, time

torch.manual_seed(42)
np.random.seed(42)

# ── 2D Parameters ────────────────────────────────────────────
N2D       = 150       # number of irregular mesh nodes
T2D       = 30        # time steps
C2D       = 1.0       # wave speed
HIDDEN2D  = 64
FREQS2D   = 16
EPOCHS2D  = 36000

print("=" * 65)
print("MODULE 1: 2D Membrane Wave on Irregular Mesh")
print("=" * 65)

# ── Build irregular mesh ────────────────────────────
def build_irregular_mesh(N, seed=42):
    """
    Scatter N points in [0,1]^2 with denser sampling near centre.
    Triangulate with Delaunay. Returns nodes and edges.
    """
    rng = np.random.RandomState(seed)

    # Mix: uniform + clustered near centre
    n_uniform  = N * 2 // 3
    n_cluster  = N - n_uniform
    pts_uniform = rng.uniform(0, 1, (n_uniform, 2))
    pts_cluster = rng.normal([0.5, 0.5], 0.15, (n_cluster, 2))
    pts_cluster = np.clip(pts_cluster, 0.02, 0.98)
    pts = np.vstack([pts_uniform, pts_cluster])

    # Add boundary points for clean BCs
    bnd = np.array([
        [0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],
        [0,0.25],[0,0.5],[0,0.75],[0,1],
        [1,0.25],[1,0.5],[1,0.75],[1,1],
        [0.25,1],[0.5,1],[0.75,1],
    ])
    pts = np.vstack([pts, bnd])

    # Delaunay triangulation
    tri  = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = simplex[i], simplex[j]
                edges.add((min(a,b), max(a,b)))

    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Boundary mask
    tol = 0.02
    bnd_mask = (
        (pts[:,0] < tol) | (pts[:,0] > 1-tol) |
        (pts[:,1] < tol) | (pts[:,1] > 1-tol)
    )
    return pts, edge_index, bnd_mask

nodes_2d, edge_index_2d, bnd_mask_2d = build_irregular_mesh(N2D)
N_actual = len(nodes_2d)
print(f"  Mesh: {N_actual} nodes, {edge_index_2d.shape[1]//2} edges")

# Check connectivity
deg = torch.zeros(N_actual, dtype=torch.long)
deg.scatter_add_(0, edge_index_2d[0],
                 torch.ones(edge_index_2d.shape[1], dtype=torch.long))
print(f"  Node degree: min={deg.min().item()}, "
      f"max={deg.max().item()}, "
      f"mean={deg.float().mean().item():.1f}")

# ── Exact 2D solution: lowest mode ──────────────────────────
def exact_2d(x, y, t, c=C2D):
    """u(x,y,t) = sin(πx)sin(πy)cos(π√2·c·t)"""
    return (np.sin(np.pi*x) * np.sin(np.pi*y) *
            np.cos(np.pi * np.sqrt(2) * c * t))

x2d = nodes_2d[:, 0].astype(np.float32)
y2d = nodes_2d[:, 1].astype(np.float32)

# ── Fourier encoder for 2D ───────────────────────────────────
class FourierEncoder2D(nn.Module):
    def __init__(self, num_freqs=FREQS2D, hidden=HIDDEN2D):
        super().__init__()
        B = torch.randn(3, num_freqs) * np.pi   # 3 inputs: x,y,t
        self.register_buffer('B', B)
        self.net = nn.Sequential(
            nn.Linear(num_freqs*2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),      nn.Tanh(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        proj = xyt @ self.B
        feat = torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)
        return self.net(feat)

# ── GNN layer for 2D ────────────────────────────────────────
class GNNLayer2D(MessagePassing):
    def __init__(self, hidden=HIDDEN2D):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.Tanh(),
            nn.Linear(hidden,   hidden), nn.Tanh(),
        )
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

# ── 2D GNN-PINN ──────────────────────────────────────────────
class GNNPINN2D(nn.Module):
    def __init__(self, hidden=HIDDEN2D, n_layers=4):
        super().__init__()
        self.input_scale = nn.Parameter(torch.ones(3))
        self.encoder  = FourierEncoder2D(FREQS2D, hidden)
        self.gnn      = nn.ModuleList(
            [GNNLayer2D(hidden) for _ in range(n_layers)])
        self.decoder  = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.Tanh(),
            nn.Linear(hidden//2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t, edge_index):
        xyt = torch.stack([x, y, t], dim=1) * self.input_scale
        h   = self.encoder(xyt)
        for layer in self.gnn:
            h = h + layer(h, edge_index)
        return self.decoder(h).squeeze(-1)

# ── 2D MLP-PINN ──────────────────────────────────────────────
class MLPPINN2D(nn.Module):
    def __init__(self, hidden=HIDDEN2D, depth=6):
        super().__init__()
        B = torch.randn(3, FREQS2D) * np.pi
        self.register_buffer('B', B)
        self.input_scale = nn.Parameter(torch.ones(3))
        layers = [nn.Linear(FREQS2D*2, hidden), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t, edge_index=None):
        xyt  = torch.stack([x, y, t], dim=1) * self.input_scale
        proj = xyt @ self.B
        feat = torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)
        return self.net(feat).squeeze(-1)

# ── 2D Loss functions ────────────────────────────────────────
def pde_loss_2d(model, x, y, t, edge_index, c=C2D):
    """∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)"""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u    = model(x, y, t, edge_index)
    u_t  = torch.autograd.grad(u.sum(),   t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(),   x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y  = torch.autograd.grad(u.sum(),   y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    return (u_tt - c**2*(u_xx + u_yy)).pow(2).mean()

def bc_loss_2d(model, x, y, t, bnd_mask, edge_index):
    xb = torch.tensor(x[bnd_mask], dtype=torch.float32)
    yb = torch.tensor(y[bnd_mask], dtype=torch.float32)
    e  = torch.tensor([[0],[0]], dtype=torch.long)
    n  = len(xb)
    u_b = model(xb, yb, t[:n], e.expand(2, n))
    return u_b.pow(2).mean()

def ic_loss_2d(model, x, y, edge_index):
    N   = len(x)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    t0  = torch.zeros(N, requires_grad=True)
    u0  = model(x_t, y_t, t0, edge_index)
    u0_true = torch.tensor(
        np.sin(np.pi*x)*np.sin(np.pi*y),
        dtype=torch.float32)
    L_d = (u0 - u0_true).pow(2).mean()
    u0_t = torch.autograd.grad(
        u0.sum(), t0, create_graph=True)[0]
    return L_d + u0_t.pow(2).mean()

# ── Training function for 2D ────────────────────────────────
def train_2d(model, model_name, epochs=EPOCHS2D):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    x_t = torch.tensor(x2d, dtype=torch.float32)
    y_t = torch.tensor(y2d, dtype=torch.float32)

    losses, grad_norms = [], []
    log_every = epochs // 5

    print(f"\n  Training {model_name} ({epochs} epochs)...")
    t0 = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        t_col = torch.rand(N_actual) * 1.0

        Lp = pde_loss_2d(model, x_t, y_t, t_col,
                         edge_index_2d)
        Lb = bc_loss_2d(model, x2d, y2d, t_col,
                        bnd_mask_2d, edge_index_2d)
        Li = ic_loss_2d(model, x2d, y2d, edge_index_2d)
        loss = Lp + 20*Lb + 30*Li
        loss.backward()

        # Record gradient norm for Module 2
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()**2
        grad_norms.append(total_norm**0.5)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"     Ep {epoch:>5} | Loss: {loss.item():.5f} "
                  f"| PDE: {Lp.item():.5f} "
                  f"| Elapsed: {elapsed:.0f}s")

    return losses, grad_norms

# ── Train both 2D models ────────────────────────────────────
gnn_2d = GNNPINN2D()
mlp_2d = MLPPINN2D()

losses_gnn_2d, gnorms_gnn_2d = train_2d(gnn_2d, "GNN-PINN 2D")
losses_mlp_2d, gnorms_mlp_2d = train_2d(mlp_2d, "MLP-PINN 2D")

# ── Evaluate 2D models ───────────────────────────────────────
T_eval  = np.linspace(0, 1, T2D)
u_exact_2d = np.array([
    exact_2d(x2d, y2d, t) for t in T_eval])  # [T, N]

def eval_2d(model, model_name):
    model.eval()
    x_t = torch.tensor(x2d, dtype=torch.float32)
    y_t = torch.tensor(y2d, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for t_val in T_eval:
            t_t = torch.full((N_actual,), t_val,
                             dtype=torch.float32)
            u = model(x_t, y_t, t_t, edge_index_2d)
            preds.append(u.numpy())
    u_pred = np.array(preds)   # [T, N]
    rel_l2 = (np.linalg.norm(u_pred - u_exact_2d) /
              np.linalg.norm(u_exact_2d))
    print(f"  {model_name} 2D Rel-L2: {rel_l2:.5f}")
    return u_pred, rel_l2

u_gnn_2d, rl2_gnn_2d = eval_2d(gnn_2d, "GNN-PINN")
u_mlp_2d, rl2_mlp_2d = eval_2d(mlp_2d, "MLP-PINN")

# ── 2D Result Plot ───────────────────────────────────────────
t_snap_idx = T2D // 2   # t = 0.5
u_ref_snap = u_exact_2d[t_snap_idx]
u_gnn_snap = u_gnn_2d[t_snap_idx]
u_mlp_snap = u_mlp_2d[t_snap_idx]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
scatter_kw = dict(s=40, cmap='RdBu_r')
vmin, vmax = u_ref_snap.min(), u_ref_snap.max()

sc0 = axes[0].scatter(x2d, y2d, c=u_ref_snap,
                      vmin=vmin, vmax=vmax, **scatter_kw)
axes[0].set_title("Reference (exact)", fontsize=11)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(sc0, ax=axes[0])

sc1 = axes[1].scatter(x2d, y2d, c=u_gnn_snap,
                      vmin=vmin, vmax=vmax, **scatter_kw)
axes[1].set_title(
    f"GNN-PINN 2D\nRel-L2={rl2_gnn_2d:.4f}", fontsize=11)
axes[1].set_xlabel("x")
plt.colorbar(sc1, ax=axes[1])

sc2 = axes[2].scatter(x2d, y2d, c=u_mlp_snap,
                      vmin=vmin, vmax=vmax, **scatter_kw)
axes[2].set_title(
    f"MLP-PINN 2D\nRel-L2={rl2_mlp_2d:.4f}", fontsize=11)
axes[2].set_xlabel("x")
plt.colorbar(sc2, ax=axes[2])

# Also show the irregular mesh
axes[3].triplot(x2d, y2d,
                Delaunay(nodes_2d).simplices,
                color='gray', lw=0.3, alpha=0.6)
axes[3].scatter(x2d[bnd_mask_2d],
                y2d[bnd_mask_2d],
                c='red', s=20, label='Boundary')
axes[3].scatter(x2d[~bnd_mask_2d],
                y2d[~bnd_mask_2d],
                c='blue', s=10, alpha=0.4, label='Interior')
axes[3].set_title(
    f"Irregular Mesh\n{N_actual} nodes", fontsize=11)
axes[3].set_xlabel("x"); axes[3].legend(fontsize=7)

plt.suptitle(
    f"2D Membrane Wave on Irregular Mesh  (t=0.5)\n"
    f"GNN Rel-L2={rl2_gnn_2d:.4f}  |  "
    f"MLP Rel-L2={rl2_mlp_2d:.4f}",
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("2d_membrane_result.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved: 2d_membrane_result.png")

np.save("results_2d.npy",
        {'gnn_rel_l2': rl2_gnn_2d,
         'mlp_rel_l2': rl2_mlp_2d,
         'losses_gnn': losses_gnn_2d,
         'losses_mlp': losses_mlp_2d,
         'gnorms_gnn': gnorms_gnn_2d,
         'gnorms_mlp': gnorms_mlp_2d},
        allow_pickle=True)


# ============================================================
# █ Module 2
# TRAINING DIAGNOSTICS
# Loss curves + gradient norm plots
# ============================================================

print("\n" + "=" * 65)
print("MODULE 2: Training Diagnostics")
print("=" * 65)

def plot_diagnostics(losses_gnn, losses_mlp,
                     gnorms_gnn, gnorms_mlp,
                     title_suffix="1D Bimetallic",
                     fname="diagnostics_1d.png"):
    """
    Generate 4-panel diagnostic figure:
      Panel 1: Loss curves (log scale)
      Panel 2: Gradient norms (log scale)
      Panel 3: Loss ratio GNN/MLP
      Panel 4: Gradient norm ratio
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    epochs_g = np.arange(1, len(losses_gnn)+1)
    epochs_m = np.arange(1, len(losses_mlp)+1)

    # Panel 1 — Loss curves
    axes[0,0].semilogy(epochs_g, losses_gnn,
                       'b-', lw=0.8, label='GNN-PINN',
                       alpha=0.8)
    axes[0,0].semilogy(epochs_m, losses_mlp,
                       'g-', lw=0.8, label='MLP-PINN',
                       alpha=0.8)
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Total Loss (log scale)")
    axes[0,0].set_title(f"Training Loss — {title_suffix}")
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # Panel 2 — Gradient norms
    axes[0,1].semilogy(epochs_g, gnorms_gnn,
                       'b-', lw=0.8, label='GNN-PINN',
                       alpha=0.8)
    axes[0,1].semilogy(epochs_m, gnorms_mlp,
                       'g-', lw=0.8, label='MLP-PINN',
                       alpha=0.8)
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Gradient L2 Norm (log scale)")
    axes[0,1].set_title(f"Gradient Norm — {title_suffix}")
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    # Panel 3 — Loss ratio (smoothed)
    min_len = min(len(losses_gnn), len(losses_mlp))
    ratio_l = (np.array(losses_gnn[:min_len]) /
               (np.array(losses_mlp[:min_len]) + 1e-10))
    window  = max(min_len//20, 1)
    ratio_smooth = np.convolve(
        ratio_l, np.ones(window)/window, mode='valid')
    axes[1,0].plot(ratio_smooth, 'r-', lw=1.2)
    axes[1,0].axhline(1.0, color='gray', ls='--', lw=1)
    axes[1,0].fill_between(
        range(len(ratio_smooth)), 1, ratio_smooth,
        where=ratio_smooth>1,
        alpha=0.15, color='blue', label='GNN worse')
    axes[1,0].fill_between(
        range(len(ratio_smooth)), 1, ratio_smooth,
        where=ratio_smooth<1,
        alpha=0.15, color='green', label='MLP worse')
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("Loss ratio GNN/MLP")
    axes[1,0].set_title("Loss Ratio (>1 = GNN losing)")
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(alpha=0.3)

    # Panel 4 — Gradient norm ratio
    min_len2 = min(len(gnorms_gnn), len(gnorms_mlp))
    ratio_g  = (np.array(gnorms_gnn[:min_len2]) /
                (np.array(gnorms_mlp[:min_len2]) + 1e-10))
    ratio_gs = np.convolve(
        ratio_g, np.ones(window)/window, mode='valid')
    axes[1,1].plot(ratio_gs, 'purple', lw=1.2)
    axes[1,1].axhline(1.0, color='gray', ls='--', lw=1)
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("Grad norm ratio GNN/MLP")
    axes[1,1].set_title(
        "Gradient Norm Ratio\n"
        "(>1 = GNN has larger/noisier gradients)")
    axes[1,1].grid(alpha=0.3)

    plt.suptitle(
        f"Training Diagnostics — {title_suffix}",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")

# Generate diagnostic plots for 2D results
plot_diagnostics(
    losses_gnn_2d, losses_mlp_2d,
    gnorms_gnn_2d, gnorms_mlp_2d,
    title_suffix="2D Membrane",
    fname="diagnostics_2d.png")

print("  Add gradient norm recording to your 1D training loop")
print("  by appending the following inside the epoch loop:")
print("""
    # Inside epoch loop — add after loss.backward():
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()**2
    grad_norms.append(total_norm**0.5)
""")


# ============================================================
# █ Module 3
# ALTERNATIVE GNN AGGREGATION STRATEGIES
# Compare: Mean / Max / Attention (GAT) / Sum
# ============================================================

print("\n" + "=" * 65)
print("MODULE 3: Alternative GNN Aggregation Strategies")
print("=" * 65)

# ── Aggregation variants ─────────────────────────────────────
class MeanAggLayer(MessagePassing):
    """Standard mean aggregation (your current model)"""
    def __init__(self, hidden):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh())
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class MaxAggLayer(MessagePassing):
    """Max aggregation — captures extremes, less smoothing"""
    def __init__(self, hidden):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh())
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class SumAggLayer(MessagePassing):
    """Sum aggregation — degree-sensitive"""
    def __init__(self, hidden):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh())
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class AttentionAggLayer(nn.Module):
    """
    Graph Attention (GAT) — learns which neighbours matter.
    Most expressive aggregation. Uses PyG GATConv.
    """
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.gat = GATConv(hidden, hidden//heads,
                           heads=heads, concat=True,
                           dropout=0.0)
        self.act = nn.Tanh()

    def forward(self, x, edge_index):
        return self.act(self.gat(x, edge_index))

# ── Generic GNN-PINN with swappable aggregation ──────────────
class GNNPINNVariant(nn.Module):
    def __init__(self, agg_type='mean',
                 hidden=HIDDEN2D, n_layers=4,
                 num_freqs=FREQS2D):
        super().__init__()
        self.agg_type    = agg_type
        self.input_scale = nn.Parameter(torch.ones(3))
        B = torch.randn(3, num_freqs) * np.pi
        self.register_buffer('B', B)
        self.encoder_net = nn.Sequential(
            nn.Linear(num_freqs*2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),      nn.Tanh())

        if agg_type == 'mean':
            LayerClass = MeanAggLayer
        elif agg_type == 'max':
            LayerClass = MaxAggLayer
        elif agg_type == 'sum':
            LayerClass = SumAggLayer
        elif agg_type == 'attention':
            LayerClass = AttentionAggLayer
        else:
            raise ValueError(f"Unknown agg_type: {agg_type}")

        if agg_type == 'attention':
            self.gnn = nn.ModuleList(
                [AttentionAggLayer(hidden) for _ in range(n_layers)])
        else:
            self.gnn = nn.ModuleList(
                [LayerClass(hidden) for _ in range(n_layers)])

        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.Tanh(),
            nn.Linear(hidden//2, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, xyt):
        xyt = xyt * self.input_scale
        proj = xyt @ self.B
        feat = torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)
        return self.encoder_net(feat)

    def forward(self, x, y, t, edge_index):
        xyt = torch.stack([x, y, t], dim=1)
        h   = self.encode(xyt)
        for layer in self.gnn:
            h = h + layer(h, edge_index)
        return self.decoder(h).squeeze(-1)

# ── Train all aggregation variants ──────────────────────────
AGG_TYPES  = ['mean', 'max', 'sum', 'attention']
AGG_LABELS = ['Mean (current)', 'Max', 'Sum', 'Attention (GAT)']
AGG_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
EPOCHS_AGG = 2000   # shorter for comparison

agg_results = {}

for agg, label in zip(AGG_TYPES, AGG_LABELS):
    print(f"\n  Training {label}...")
    model_agg = GNNPINNVariant(agg_type=agg)
    optimizer  = torch.optim.Adam(
        model_agg.parameters(), lr=5e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_AGG, eta_min=1e-5)

    x_t = torch.tensor(x2d, dtype=torch.float32)
    y_t = torch.tensor(y2d, dtype=torch.float32)
    losses_agg, gnorms_agg = [], []

    for epoch in range(1, EPOCHS_AGG+1):
        model_agg.train()
        optimizer.zero_grad()
        t_col = torch.rand(N_actual)
        Lp = pde_loss_2d(model_agg, x_t, y_t,
                         t_col, edge_index_2d)
        Lb = bc_loss_2d(model_agg, x2d, y2d,
                        t_col, bnd_mask_2d, edge_index_2d)
        Li = ic_loss_2d(model_agg, x2d, y2d, edge_index_2d)
        loss = Lp + 20*Lb + 30*Li
        loss.backward()

        total_norm = sum(
            p.grad.data.norm(2).item()**2
            for p in model_agg.parameters()
            if p.grad is not None) ** 0.5
        gnorms_agg.append(total_norm)

        torch.nn.utils.clip_grad_norm_(
            model_agg.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses_agg.append(loss.item())

    # Evaluate
    model_agg.eval()
    preds = []
    with torch.no_grad():
        for t_val in T_eval:
            t_t = torch.full((N_actual,), t_val,
                             dtype=torch.float32)
            u = model_agg(x_t, y_t, t_t, edge_index_2d)
            preds.append(u.numpy())
    u_agg = np.array(preds)
    rl2   = (np.linalg.norm(u_agg - u_exact_2d) /
             np.linalg.norm(u_exact_2d))
    print(f"    {label}: Rel-L2 = {rl2:.5f}")

    agg_results[agg] = {
        'rel_l2': rl2, 'losses': losses_agg,
        'gnorms': gnorms_agg, 'label': label}

# ── Aggregation comparison plots ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1 — Loss curves
for agg, col in zip(AGG_TYPES, AGG_COLORS):
    r = agg_results[agg]
    axes[0].semilogy(r['losses'], color=col,
                     lw=0.8, label=r['label'], alpha=0.85)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (log scale)")
axes[0].set_title("Training Loss by Aggregation Type")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Panel 2 — Gradient norms
for agg, col in zip(AGG_TYPES, AGG_COLORS):
    r = agg_results[agg]
    axes[1].semilogy(r['gnorms'], color=col,
                     lw=0.8, label=r['label'], alpha=0.85)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Gradient Norm (log scale)")
axes[1].set_title("Gradient Norms by Aggregation Type")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Panel 3 — Final Rel-L2 bar chart
labels = [agg_results[a]['label'].replace(' ','\n')
          for a in AGG_TYPES]
vals   = [agg_results[a]['rel_l2'] for a in AGG_TYPES]
bars   = axes[2].bar(labels, vals, color=AGG_COLORS,
                     edgecolor='black', linewidth=0.5)
# Add MLP-PINN reference line
axes[2].axhline(rl2_mlp_2d, color='black',
                ls='--', lw=1.5,
                label=f'MLP-PINN ({rl2_mlp_2d:.4f})')
for bar, val in zip(bars, vals):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 val + 0.002, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=8)
axes[2].set_ylabel("Rel-L2 Error")
axes[2].set_title("Final Accuracy by Aggregation\n"
                  "(dashed = MLP-PINN baseline)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3, axis='y')

plt.suptitle(
    "GNN Aggregation Strategy Comparison — 2D Membrane",
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("aggregation_comparison.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved: aggregation_comparison.png")

# ── Print summary table ───────────────────────────────────────
print("\n" + "="*55)
print("  AGGREGATION STRATEGY SUMMARY TABLE")
print("="*55)
print(f"  {'Method':<25} {'Rel-L2':>10}")
print(f"  {'-'*40}")
print(f"  {'MLP-PINN (baseline)':<25} {rl2_mlp_2d:>10.5f}")
for agg in AGG_TYPES:
    r = agg_results[agg]
    better = " ✅" if r['rel_l2'] < rl2_mlp_2d else " ❌"
    print(f"  {('GNN-'+r['label']):<25} "
          f"{r['rel_l2']:>10.5f}{better}")
print("="*55)
print("  ✅ = GNN variant beats MLP baseline")
print("  ❌ = MLP baseline still wins")

# ── Save all results ─────────────────────────────────────────
np.save("agg_results.npy", agg_results, allow_pickle=True)
print("\nSaved: agg_results.npy")

print("\n" + "="*65)
print("ALL MODULES COMPLETE")
print("=" * 65)