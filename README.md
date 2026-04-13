<div align="center">

# 🧱 Physics-Informed Neural Network for 2D Linear Elasticity

**Predicting displacement and stress fields in steel — without FEM labels**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-PINN-FF6F00?logo=tensorflow&logoColor=white)](.)
[![Best Val Loss](https://img.shields.io/badge/Best_Val_Loss-2.22-brightgreen)](.)

</div>

---

## The Idea

Finite Element Methods (FEM) produce gold-standard solutions for structural mechanics, but they require dense meshes, expensive solvers, and domain-specific software. For rapid design sweeps — where one needs approximate field predictions across many configurations quickly — that overhead becomes a bottleneck.

Physics-Informed Neural Networks (PINNs) offer an alternative: embed the governing PDEs directly into the loss function of a neural network, train on sparsely sampled collocation points rather than labeled FEM output, and obtain continuous field predictions across the entire domain. The network learns not from data but from *physics*.

This project applies that idea to a concrete problem: predicting the **displacement field, stress components, and boundary condition satisfaction** of a steel plate under load — using only the Navier–Cauchy equations, Hooke's law, and boundary conditions as supervision.

## Problem Setup

```
                    4 N uniform vertical load (↓)
               ┌─────────────────────────────────┐
               │                                 │
               │          1m × 1m steel          │
               │            plate                │
               │                                 │
               │       E = 210 GPa               │
               │       ν = 0.3                   │
               │                                 │
               ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                    Fixed (Dirichlet BC: u = 0)
```

A 1 m × 1 m steel plate, fixed at the bottom edge (zero displacement), subjected to a uniform 4 N vertical traction on the top edge (Neumann BC). The left and right edges are traction-free. Material properties correspond to structural steel (E = 210 GPa, ν = 0.3).

The governing physics:

- **Navier–Cauchy equations** — equilibrium of the displacement field under body and surface forces
- **Hooke's law** (plane stress) — constitutive relation linking strain to stress
- **Dirichlet BC** — zero displacement at the fixed bottom boundary
- **Neumann BC** — prescribed traction at the loaded top boundary

Hand derivations for the Neumann boundary conditions and body forces are included in the repository ([`Neumann_Boundary_Condition_Derivation.pdf`](Neumann_Boundary_Condition_Derivation.pdf), [`Body Force derivation.docx`](Body%20Force%20derivation.docx)).

## Methodology

### Collocation Sampling

Rather than meshing the domain, **Latin Hypercube Sampling (LHS)** generates collocation points where the PDE residuals are evaluated:

- **1,000 interior points** — where the Navier–Cauchy equilibrium must hold
- **50 points per boundary edge** (200 total) — where Dirichlet or Neumann conditions are enforced

This quasi-random distribution ensures good coverage of the domain without the cost of a structured mesh.

### Loss Function

The total loss is a weighted sum of three physically meaningful terms:

```
L_total = L_pde  +  L_bc  +  L_stress
           │          │         │
           │          │         └── Hooke's law consistency
           │          └──────────── Dirichlet + Neumann residuals
           └─────────────────────── Navier–Cauchy PDE residual
```

Each term drives the network toward a different aspect of physical correctness. There are no supervised labels — the loss is entirely physics-derived.

### Hyperparameter Sweep

A **27-configuration grid search** explored:

| Hyperparameter | Values Tested |
|---|---|
| Hidden layers | 4, 6, 8 |
| Neurons per layer | 10, 15, 20 |
| Dropout rate | 0.1, 0.2, 0.3 |

All configurations used **Glorot (Xavier) initialization** and were tracked with validation loss monitoring.

## Results

### Best Architecture

| Parameter | Value |
|---|---|
| Hidden layers | 6 |
| Neurons per layer | 15 |
| Dropout | 0.3 |
| Initialization | Glorot |
| **Validation loss** | **2.22** |

The **6×15** topology with dropout 0.3 produced:

- **Smooth displacement fields** — continuous u(x, y) and v(x, y) predictions across the full domain, free of mesh artifacts
- **Physically coherent stress distribution** — σ_xx, σ_yy, and τ_xy fields that respect equilibrium and constitutive relations
- **Satisfied boundary conditions** — near-zero displacement at the fixed edge, correct traction at the loaded edge

Contour plots of the predicted fields are available in [`contours/`](contours/).

### What the Loss Means

The validation loss of 2.22 is a composite PDE + BC + stress residual — not a supervised error metric. It quantifies how well the network obeys the governing equations at unseen collocation points. A clear next step is computing **R² against FEM ground truth** to benchmark absolute accuracy.

## Repository Structure

```
PINN_2D_Elasticity/
├── src/                # Core PINN implementation
├── solve/              # Solver logic — forward pass + loss computation
├── optimize/           # Hyperparameter sweep (27 configs)
├── data/               # Collocation point datasets (LHS-generated)
├── contours/           # Predicted field contour plots
├── Notebook/           # Jupyter notebooks for exploration & visualization
├── utils/              # Shared utilities
├── env/                # Environment configuration
├── Body Force derivation.docx          # Hand derivation of body forces
├── Neumann_Boundary_Condition_Derivation.pdf  # Hand derivation of Neumann BCs
├── solidmechanics_model_stack.sav      # Saved best model
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| **PINN Framework** | TensorFlow (custom training loop with `GradientTape`) |
| **Sampling** | Latin Hypercube Sampling (LHS) |
| **Optimization** | Adam optimizer + grid search over 27 configurations |
| **Visualization** | Matplotlib (contour plots of displacement & stress fields) |
| **Physics** | Navier–Cauchy equations, Hooke's law (plane stress) |

## Possible Extensions

- **Adaptive collocation sampling** — concentrate points where PDE residuals are highest, improving accuracy without increasing total point count
- **Nonlinear material models** — extend beyond Hooke's law to plasticity or hyperelasticity
- **R² vs. FEM benchmark** — compare PINN predictions against a high-fidelity FEM solution for quantitative validation
- **Transfer learning** — pre-train on one geometry, fine-tune on variations for rapid design sweeps

---

<div align="center">

📬 sayeed.shahriar@gmail.com · [Portfolio](https://submerged-in-matrix.github.io/projects/pinn-elasticity/) · [GitHub](https://github.com/submerged-in-matrix)

</div>
