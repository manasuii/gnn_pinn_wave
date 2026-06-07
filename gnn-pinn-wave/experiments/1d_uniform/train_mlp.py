"""
experiments/1d_uniform/train_mlp.py
=====================================================================
Evaluates the trained MLP-PINN on the UNIFORM rod.

IMPORTANT: Run mlp_pinn_baseline.py FIRST to train the model.
That script saves: mlp_model_FINAL.pt

This script:
  1. Loads the trained MLP-PINN weights
  2. Evaluates on the uniform rod (c=1.0, exact solution known)
  3. Prints Rel-L2 error
  4. Saves comparison plot

Place this file in: experiments/1d_uniform/train_mlp.py
Run from project root: python experiments/1d_uniform/train_mlp.py
"""

import sys
import os

# ── Allow imports from models/ ───────────────────────────────
# Adjust path so we can import from models/
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '..', '..', 'models'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mlp_pinn import MLPPINN1D
from layers import build_chain_edges, tile_edges

# ============================================================
# 1.  PARAMETERS  (must match mlp_pinn_baseline.py)
# ============================================================
N         = 40
T         = 60
HIDDEN    = 128
DEPTH     = 8
NUM_FREQS = 24

CHECKPOINT = 'mlp_model_FINAL.pt'

# Check checkpoint exists
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"\n❌ Cannot find '{CHECKPOINT}'\n"
        f"Run mlp_pinn_baseline.py first to train and save the model.\n"
        f"Expected file: {os.path.abspath(CHECKPOINT)}")

# ============================================================
# 2.  LOAD MODEL
# ============================================================
model = MLPPINN1D(hidden=HIDDEN, depth=DEPTH,
                  num_freqs=NUM_FREQS)
model.load_state_dict(torch.load(CHECKPOINT,
                                  map_location='cpu'))
model.eval()
print(f"✅ Loaded: {CHECKPOINT}")
n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {n_params:,}")

# ============================================================
# 3.  BUILD EVALUATION GRID
# ============================================================
x_nodes = torch.linspace(0, 1, N)
t_vals  = torch.linspace(0, 1, T)

x_col = x_nodes.repeat(T)            # [N*T]
t_col = t_vals.repeat_interleave(N)  # [N*T]

# ============================================================
# 4.  PREDICT
# ============================================================
with torch.no_grad():
    u_pred = model(x_col, t_col, None).reshape(T, N).numpy()

# ============================================================
# 5.  EXACT SOLUTION FOR UNIFORM ROD
#     u(x,t) = sin(πx) · cos(πt)   [c=1.0]
# ============================================================
X, Tv = np.meshgrid(x_nodes.numpy(), t_vals.numpy())
u_exact_uniform = np.sin(np.pi * X) * np.cos(np.pi * Tv)

# ============================================================
# 6.  METRICS
# ============================================================
abs_err        = np.abs(u_pred - u_exact_uniform)
rel_l2_uniform = (np.linalg.norm(u_pred - u_exact_uniform) /
                  np.linalg.norm(u_exact_uniform))
max_err        = abs_err.max()

print(f"\n{'='*55}")
print(f"  MLP-PINN — Uniform Rod Results")
print(f"{'='*55}")
print(f"  Rel-L2 error : {rel_l2_uniform:.5f}")
print(f"  Max abs error: {max_err:.5f}")
print(f"{'='*55}")
print(f"\n  → Use this number in your paper Table 1:")
print(f"     MLP-PINN | 1D Uniform | {rel_l2_uniform:.4f}")

# ============================================================
# 7.  PLOT
# ============================================================
x_np  = x_nodes.numpy()
t_np  = t_vals.numpy()
vmin  = u_exact_uniform.min()
vmax  = u_exact_uniform.max()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kw = dict(aspect='auto', origin='lower',
          extent=[0,1,0,1], cmap='RdBu_r',
          vmin=vmin, vmax=vmax)

im0 = axes[0].imshow(u_exact_uniform, **kw)
axes[0].set_title("Exact: sin(πx)cos(πt)", fontsize=11)
axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(u_pred, **kw)
axes[1].set_title("MLP-PINN Prediction", fontsize=11)
axes[1].set_xlabel("x")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(abs_err, aspect='auto', origin='lower',
                     extent=[0,1,0,1], cmap='plasma')
axes[2].set_title(
    f"Absolute Error  (Rel-L2={rel_l2_uniform:.4f})",
    fontsize=11)
axes[2].set_xlabel("x")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(
    "MLP-PINN: 1D Uniform Rod  "
    "$u = \\sin(\\pi x)\\cos(\\pi t)$",
    fontsize=13, fontweight='bold')
plt.tight_layout()

out = os.path.join('results', 'mlp_uniform_result.png')
os.makedirs('results', exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out}")

# Snapshot comparison
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
snaps = [0.0, 0.25, 0.5, 1.0]
for ax, ts in zip(axes2, snaps):
    ti = np.argmin(np.abs(t_np - ts))
    ax.plot(x_np, u_exact_uniform[ti], 'r--',
            lw=2, label='Exact')
    ax.plot(x_np, u_pred[ti],          'g-',
            lw=2, label='MLP-PINN')
    ax.set_title(f"t = {ts:.2f}")
    ax.set_xlabel("x")
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes2[0].set_ylabel("u(x, t)")
plt.suptitle(
    f"MLP-PINN Snapshots — Uniform Rod  "
    f"Rel-L2={rel_l2_uniform:.4f}",
    fontsize=12)
plt.tight_layout()
out2 = os.path.join('results', 'mlp_uniform_snapshots.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out2}")