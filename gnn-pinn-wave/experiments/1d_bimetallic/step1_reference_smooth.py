"""
step1_reference_smooth.py
=====================================================================
Generates numerical ground truth for the SMOOTH bimetallic rod.
Must be run before wave_gnn_pinn_v5_smooth.py.

Uses method of lines + scipy RK45.
c(x) = c1 + (c2-c1) * sigmoid((x-0.5)/width)

OUTPUT
──────
  u_ref_smooth.npy
  x_nodes_smooth.npy
  t_eval_smooth.npy
  reference_smooth.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit     # stable sigmoid

# ── Parameters ───────────────────────────────────────────────
N         = 200
T_EVAL    = 100
C1        = 1.0
C2        = 1.5
CENTER    = 0.5
WIDTH     = 0.03      # must match v5 hyperparameter

x_nodes   = np.linspace(0, 1, N)
t_eval    = np.linspace(0, 1, T_EVAL)
dx        = x_nodes[1] - x_nodes[0]

# ── Smooth wave speed field ───────────────────────────────────
c_field   = C1 + (C2 - C1) * expit((x_nodes - CENTER) / WIDTH)

print("Smooth bimetallic reference solution")
print(f"  N={N}, T={T_EVAL}, dx={dx:.5f}")
print(f"  c1={C1}, c2={C2}, center={CENTER}, width={WIDTH}")
print(f"  c at x=0.0: {c_field[0]:.4f}")
print(f"  c at x=0.5: {c_field[N//2]:.4f}")
print(f"  c at x=1.0: {c_field[-1]:.4f}")

# ── Initial conditions ────────────────────────────────────────
u0     = np.sin(np.pi * x_nodes)
v0     = np.zeros(N)
state0 = np.concatenate([u0, v0])

# ── Method of lines RHS ──────────────────────────────────────
def wave_rhs(t, state):
    u    = state[:N]
    v    = state[N:]
    u_xx = np.zeros(N)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0]    = 0.0
    u_xx[-1]   = 0.0
    dudt = v
    dvdt = c_field**2 * u_xx
    return np.concatenate([dudt, dvdt])

# ── Solve ─────────────────────────────────────────────────────
print("\nSolving... (~10-30 seconds)")
sol = solve_ivp(
    wave_rhs,
    t_span  = [0.0, 1.0],
    y0      = state0,
    t_eval  = t_eval,
    method  = 'RK45',
    rtol    = 1e-8,
    atol    = 1e-8,
)

if not sol.success:
    raise RuntimeError(f"Solver failed: {sol.message}")

u_ref = sol.y[:N, :].T    # [T, N]
print(f"Solved. Shape: {u_ref.shape}")
print(f"u range: [{u_ref.min():.4f}, {u_ref.max():.4f}]")

# ── Save ──────────────────────────────────────────────────────
np.save("u_ref_smooth.npy",    u_ref)
np.save("x_nodes_smooth.npy",  x_nodes)
np.save("t_eval_smooth.npy",   t_eval)
print("\nSaved: u_ref_smooth.npy, x_nodes_smooth.npy, t_eval_smooth.npy")

# ── Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Wave speed profile
axes[0].plot(x_nodes, c_field, 'b-', lw=2)
axes[0].axvline(CENTER, color='red', ls='--', lw=1.2,
                label=f'Centre x={CENTER}')
axes[0].axhline(C1, color='gray', ls=':', lw=1, label=f'c₁={C1}')
axes[0].axhline(C2, color='gray', ls='--', lw=1, label=f'c₂={C2}')
axes[0].set_xlabel("x"); axes[0].set_ylabel("c(x)")
axes[0].set_title("Smooth Wave Speed Profile")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Heatmap
X2d, T2d = np.meshgrid(x_nodes, t_eval)
cf = axes[1].contourf(X2d, T2d, u_ref, levels=60, cmap='RdBu_r')
axes[1].axvline(x=CENTER, color='yellow', lw=2,
                ls='--', label='Interface')
axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
axes[1].set_title("Reference Solution u(x,t)")
axes[1].legend(fontsize=8)
plt.colorbar(cf, ax=axes[1])

# Snapshots
snap_times = [0.0, 0.25, 0.50, 0.75]
colors     = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
for ts, col in zip(snap_times, colors):
    idx = np.argmin(np.abs(t_eval - ts))
    axes[2].plot(x_nodes, u_ref[idx],
                 color=col, lw=1.8, label=f't={ts:.2f}')
axes[2].axvline(CENTER, color='black', lw=1.5,
                ls='--', label='Interface')
axes[2].axhline(0, color='lightgray', lw=0.5)
axes[2].set_xlabel("x"); axes[2].set_ylabel("u(x,t)")
axes[2].set_title("Snapshots")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.suptitle(
    f"Smooth Bimetallic Rod Reference  "
    f"$c_1={C1}$, $c_2={C2}$, width={WIDTH}",
    fontsize=13)
plt.tight_layout()
plt.savefig("reference_smooth.png", dpi=150, bbox_inches='tight')
plt.show()

# Sanity checks
print("\nSanity checks:")
print(f"  u(x,0) max = {u_ref[0].max():.4f}  (should be ~1.0)")
t_half = np.argmin(np.abs(t_eval - 0.5))
asym   = np.abs(u_ref[t_half, :N//2].mean() -
                u_ref[t_half, N//2:].mean())
print(f"  Asymmetry at t=0.5: {asym:.4f}  (>0 means interface effect visible)")
print(f"\n✅ Done. Now run: python wave_gnn_pinn_v5_smooth.py")