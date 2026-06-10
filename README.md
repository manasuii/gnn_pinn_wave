 GNN-PINN Wave Propagation

<div align="center">

# When Does Graph Structure Help in Physics-Informed Neural Networks?

### A Comparative Study on Elastic Wave Propagation in Heterogeneous Mechanical Structures

[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Your Name]** — [Your University], Kathmandu, Nepal

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the complete code, data, and
experiments for our paper:

> **"When Does Graph Structure Help in Physics-Informed
> Neural Networks? A Comparative Study on Elastic Wave
> Propagation in Heterogeneous Mechanical Structures"**

We study the 1D and 2D elastic wave equation:

```
1D:  ∂²u/∂t² = c(x)² · ∂²u/∂x²
2D:  ∂²u/∂t² = c²  · (∂²u/∂x² + ∂²u/∂y²)
```

and compare two neural network architectures for solving it:

| Architecture | Description |
|---|---|
| **GNN-PINN** | Physics-Informed Neural Network augmented with Graph Neural Network message-passing layers. Encodes spatial mesh topology through iterative neighbour aggregation. |
| **MLP-PINN** | Standard Physics-Informed Neural Network using a deep Multilayer Perceptron with Fourier feature encoding. No graph structure. Treats all spatial points independently. |

Both models share **identical** Fourier feature encoders,
causal time-window training, and loss functions.
The **only** difference is the presence or absence of GNN layers.
This makes the comparison strictly controlled and fair.

---

## Key Findings

Our experiments across three benchmarks reveal a clear
**problem-dependent trade-off**:

```
┌─────────────────────┬────────────────┬─────────────────┬──────────────────┐
│ Method              │ 1D Uniform rod │ 1D Bimetallic   │ 2D Irregular     │
│                     │                │ rod             │ mesh             │
├─────────────────────┼────────────────┼─────────────────┼──────────────────┤
│ MLP-PINN (baseline) │ 0.2900         │ 0.0024 ✅ BEST  │ 0.0116 ✅ BEST   │
│ GNN-PINN (ours)     │ 0.1061 ✅ BEST │ 0.2456          │ 0.8552           │
├─────────────────────┼────────────────┼─────────────────┼──────────────────┤
│ Winner              │ GNN by 2.7×    │ MLP by 102×     │ MLP by 74×       │
└─────────────────────┴────────────────┴─────────────────┴──────────────────┘
```

**Main conclusion:**
MLP-PINN with Fourier features outperforms GNN-PINN on
heterogeneous and 2D problems by large margins (74×–102×),
even with equal training budgets.
GNN only wins on the simplest 1D uniform case.

**Why this happens:**
On 1D chain graphs, GNN message passing creates *redundant
inductive bias* already captured by the PDE loss, and
introduces gradient noise from scatter aggregation that
corrupts autograd-derived PDE derivatives.
Fourier feature encoding is a more powerful inductive bias
for wave PDE problems than graph spatial connectivity.

**Practical guidance:**
- Use MLP-PINN for 1D/2D smooth wave problems
- Reserve GNN-PINN for very large irregular meshes,
  branched geometries, or multi-physics coupling

---

## Repository Structure

```
gnn-pinn-wave/
│
├── README.md                          ← you are here
├── requirements.txt                   ← all dependencies
├── LICENSE                            ← MIT license
│
├── data/                              ← reference solutions
│   ├── u_ref_smooth.npy               ← 1D bimetallic reference [100×200]
│   ├── x_nodes_smooth.npy             ← spatial coordinates [200]
│   ├── t_eval_smooth.npy              ← time coordinates [100]
│   └── README_data.md                 ← how data was generated
│
├── models/                            ← architecture definitions
│   ├── layers.py                      ← shared building blocks
│   │                                    (FourierEncoder, WaveGNNLayer,
│   │                                     MaxAggLayer, SumAggLayer,
│   │                                     AttentionAggLayer, graph utils)
│   ├── gnn_pinn.py                    ← GNN-PINN (1D and 2D)
│   └── mlp_pinn.py                    ← MLP-PINN baseline (1D and 2D)
│
├── experiments/                       ← training scripts
│   │
│   ├── 1d_bimetallic/                 ← main 1D experiment
│   │   ├── step1_reference_smooth.py  ← STEP 1: generate reference data
│   │   ├── wave_gnn_pinn_FINAL.py     ← STEP 2: train GNN-PINN
│   │   ├── mlp_pinn_baseline.py       ← STEP 3: train MLP-PINN
│   │   └── wave_gnn_pinn_RESUME.py    ← resume from checkpoint
│   │
│   ├── 1d_uniform/                    ← uniform rod evaluation
│   │   ├── gnn_pinn.py                ← GNN-PINN on uniform rod
│   │   └── train_mlp.py               ← evaluate MLP on uniform rod
│   │
│   └── 2d_membrane/                   ← 2D extension
│       └── extensions_code.py         ← 2D benchmark + diagnostics
│                                         + aggregation comparison
│
├── results/                           ← saved outputs
│   ├── figures/                       ← all paper figures (.png)
│   │   ├── bimetallic_result_FINAL.png
│   │   ├── mlp_result_FINAL.png
│   │   ├── bimetallic_snapshots_FINAL.png
│   │   ├── mlp_snapshots_FINAL.png
│   │   ├── bimetallic_reflection_FINAL.png
│   │   ├── 2d_membrane_result.png
│   │   ├── diagnostics_2d.png
│   │   └── aggregation_comparison.png
│   │
│   ├── checkpoints/                   ← saved model weights (.pt)
│   │   ├── model_FINAL.pt             ← final GNN-PINN weights
│   │   ├── mlp_model_FINAL.pt         ← final MLP-PINN weights
│   │   ├── checkpoint_window_1.pt     ← GNN per-window checkpoints
│   │   └── ...
│   │
│   └── metrics.json                   ← all Rel-L2 results
│
├── paper/
│   └── gnn_pinn_paper_COMPLETE.tex    ← LaTeX source
│
└── notebooks/
    └── reproduce_results.ipynb        ← step-by-step Jupyter notebook
```

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/[YOUR-USERNAME]/gnn-pinn-wave.git
cd gnn-pinn-wave
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Using conda
conda create -n gnn_pinn python=3.10
conda activate gnn_pinn

# OR using venv
python -m venv gnn_pinn_env
source gnn_pinn_env/bin/activate        # Linux/Mac
gnn_pinn_env\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install PyTorch Geometric

PyTorch Geometric requires a version-matched install.
Replace `cu118` with your CUDA version, or use `cpu`:

```bash
# CPU only
pip install torch-geometric
pip install torch-scatter torch-sparse -f \
    https://data.pyg.org/whl/torch-2.0.0+cpu.html

# CUDA 11.8
pip install torch-scatter torch-sparse -f \
    https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Verify installation

```python
import torch
import torch_geometric
import numpy as np
import scipy

print(f"PyTorch:          {torch.__version__}")
print(f"PyTorch Geometric:{torch_geometric.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
```

---

## Quick Start

### Reproduce the main result in 3 commands

```bash
# 1. Generate reference solution (~20 seconds)
python experiments/1d_bimetallic/step1_reference_smooth.py

# 2. Train GNN-PINN (~3-5 hrs CPU / ~1-2 hrs GPU)
python experiments/1d_bimetallic/wave_gnn_pinn_FINAL.py

# 3. Train MLP-PINN baseline (run in parallel with step 2)
python experiments/1d_bimetallic/mlp_pinn_baseline.py
```

### Run the 2D extension

```bash
# Runs all three modules sequentially:
#   Module 1: 2D membrane benchmark
#   Module 2: Training diagnostics
#   Module 3: Aggregation strategy comparison
python experiments/2d_membrane/extensions_code.py
```

### Resume from a crashed session

If training crashes, checkpoints are saved after every window.
Resume from the last saved checkpoint:

```bash
# Automatically loads checkpoint_window_4.pt and continues
python experiments/1d_bimetallic/wave_gnn_pinn_RESUME.py
```

---

## Reproducing Paper Results

Each paper figure maps to a specific script:

| Paper figure | Script | Output file |
|---|---|---|
| Fig 1 — Wave speed profile | `step1_reference_smooth.py` | `reference_smooth.png` |
| Fig 2 — GNN-PINN heatmap | `wave_gnn_pinn_FINAL.py` | `bimetallic_result_FINAL.png` |
| Fig 3 — MLP-PINN heatmap | `mlp_pinn_baseline.py` | `mlp_result_FINAL.png` |
| Fig 4 — Snapshots (both) | Both training scripts | `*_snapshots_FINAL.png` |
| Fig 5 — R & T coefficients | `wave_gnn_pinn_FINAL.py` | `bimetallic_reflection_FINAL.png` |
| Fig 6 — 2D membrane | `extensions_code.py` | `2d_membrane_result.png` |
| Fig 7 — Diagnostics | `extensions_code.py` | `diagnostics_2d.png` |
| Fig 8 — Aggregation | `extensions_code.py` | `aggregation_comparison.png` |

### Expected runtimes

| Experiment | CPU | GPU (T4) |
|---|---|---|
| Reference solution (1D) | ~20 sec | ~20 sec |
| GNN-PINN 1D (36k epochs) | ~8-12 hrs | ~2-3 hrs |
| MLP-PINN 1D (36k epochs) | ~6-10 hrs | ~1-2 hrs |
| 2D extension (all modules) | ~4-6 hrs | ~1-2 hrs |

> **Tip:** Run GNN-PINN and MLP-PINN in parallel on two
> separate terminals or Kaggle notebooks to halve total time.

---

## Model Architecture

### Shared components (both models)

```python
# Fourier feature encoding — overcomes spectral bias
# Maps (x,t) → [sin(Bx), cos(Bx)] before the network
# B ~ N(0, π²), fixed after init, not trained

# Causal time-window training (1D only)
# Progressively expands time domain:
# t ∈ [0,0.10] → [0,0.20] → ... → [0,1.00]
# Forces temporal causality — prevents trivial solutions

# Physics loss = PDE residual + BC + IC + amplitude
# Derivatives computed via torch.autograd (exact)
```

### GNN-PINN specific

```python
# Chain graph (1D): N=40 nodes, edges connect i↔i+1
# Delaunay mesh (2D): ~180 nodes, irregular triangulation
# L=6 message-passing layers with residual connections
# Aggregation: mean / max / sum / attention (GAT)

from models.gnn_pinn import GNNPINN1D, GNNPINN2D

model_1d = GNNPINN1D(hidden=128, num_layers=6,
                      num_freqs=24, agg_type='mean')
model_2d = GNNPINN2D(hidden=64,  num_layers=4,
                      num_freqs=16, agg_type='mean')
```

### MLP-PINN specific

```python
# No graph. 8-layer MLP. edge_index accepted but ignored.
# Deeper than GNN encoder to match parameter count.

from models.mlp_pinn import MLPPINN1D, MLPPINN2D

model_1d = MLPPINN1D(hidden=128, depth=8, num_freqs=24)
model_2d = MLPPINN2D(hidden=64,  depth=6, num_freqs=16)
```

---

## Results

### Main comparison table

| Method | 1D Uniform | 1D Bimetallic | 2D Irregular |
|---|---|---|---|
| MLP-PINN | 0.2900 | **0.0024** | **0.0116** |
| GNN-PINN (mean) | **0.1061** | 0.2456 | 0.8552 |
| **Winner** | GNN (2.7×) | MLP (102×) | MLP (74×) |

All experiments use 36,000 training epochs for fair comparison.
Metric: Relative L2 error = ||u_pred - u_ref|| / ||u_ref||

### GNN aggregation comparison (2D membrane)

| Aggregation | Rel-L2 | vs MLP-PINN |
|---|---|---|
| MLP-PINN baseline | 0.0116 | — |
| GNN Mean | 0.8552 | ×73.7 worse |
| GNN Max | [see paper] | — |
| GNN Sum | [see paper] | — |
| GNN Attention (GAT) | [see paper] | — |

---

## Project Dependencies

See `requirements.txt` for full list. Key dependencies:

```
torch>=1.12.0          — neural network framework
torch-geometric>=2.0.0 — graph neural network operations
numpy>=1.22.0           — numerical computation
scipy>=1.8.0            — reference solution (RK45 solver)
matplotlib>=3.5.0       — plotting and figure generation
```

---

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{[yourname]2025gnnpinn,
  title   = {When Does Graph Structure Help in
             Physics-Informed Neural Networks?
             A Comparative Study on Elastic Wave Propagation
             in Heterogeneous Mechanical Structures},
  author  = {[Your Full Name]},
  journal = {arXiv preprint arXiv:2506.XXXXX},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.XXXXX}
}
```

---

## Acknowledgements

This work builds on the following open-source projects:
- [PyTorch](https://pytorch.org/) — neural network framework
- [PyTorch Geometric](https://pyg.org/) — GNN operations
- [Raissi et al. (2019)](https://github.com/maziarraissi/PINNs)
  — original PINN implementation

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Contact

**[Your Name]**
[Your Department], [Your University]
📧 [your.email@university.edu]
🔗 [https://github.com/YOUR-USERNAME](https://github.com/YOUR-USERNAME)

*Found a bug or have a question? Please open an
[issue](https://github.com/YOUR-USERNAME/gnn-pinn-wave/issues).*
