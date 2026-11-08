# Physics-Informed Neural Network for 2D Linear Elasticity

**Can a network trained on PDE residuals alone, with no FEM labels, recover a displacement and stress field that is checked against a known analytical solution?**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-PINN-FF6F00?logo=tensorflow&logoColor=white)](.)

---

## Scope

This is a verification study on a manufactured-solution benchmark, not a model of a physical steel plate. A displacement field is prescribed, the body force that makes it an exact solution is derived analytically, and the PINN is trained to reproduce the field from the governing equations and boundary conditions only. Because the exact solution is known, the error can be measured pointwise. The benchmark follows the 2D solid-mechanics PINN example of J. Chen ([PINN-for-solid-mechanics](https://github.com/jungangc/PINN-for-solid-mechanics)); the implementation, loss structure and hyperparameter study here are independent.

## Problem Setup

Domain: unit square, Ω = [0, 1] × [0, 1].

Material: non-dimensional, E = 4/3 and ν = 1/3, which gives λ = 1 and μ = 0.5. The constitutive law uses the plane-strain form, σₓₓ = (λ+2μ)εₓₓ + λεᵧᵧ, σᵧᵧ = (λ+2μ)εᵧᵧ + λεₓₓ, σₓᵧ = 2μεₓᵧ. No physical units are attached.

Manufactured exact solution (Q = 4 sets the amplitude of the vertical displacement):

```
u_x(x, y) = cos(2πx) · sin(πy)
u_y(x, y) = sin(πx) · (Q/4) · y⁴
f = −∇·σ(u)        (body force, derived analytically)
```

Derivations: [`Body Force derivation.docx`](Body%20Force%20derivation.docx) and [`Neumann_Boundary_Condition_Derivation.pdf`](Neumann_Boundary_Condition_Derivation.pdf).

Boundary conditions (all consistent with the exact solution):

| Edge | Dirichlet | Neumann |
|---|---|---|
| Bottom (y = 0) | uₓ = uᵧ = 0 | none |
| Top (y = 1) | uₓ = 0 | σᵧᵧ = (λ+2μ) Q sin(πx) |
| Left / right (x = 0, 1) | uᵧ = 0 | σₓₓ = 0 |

The top traction is sinusoidal in x, not uniform. The shear traction σₓᵧ is not imposed on any edge.

## Method

**Formulation.** Mixed stress–displacement. One network maps (x, y) to five outputs: uₓ, uᵧ, σₓₓ, σᵧᵧ, σₓᵧ. Strains and stress derivatives are obtained by automatic differentiation (`tf.GradientTape`).

**Loss.** Four terms, each a mean absolute error, summed without weighting:

1. Equilibrium residual, ∇·σ + f = 0, on the network's stress outputs
2. Constitutive residual, network stress versus Hooke's law applied to the network's displacement gradients
3. Dirichlet residual
4. Neumann residual (σₓₓ on the sides, σᵧᵧ on the top)

**Sampling.** Latin Hypercube Sampling: 1,000 interior collocation points and 50 points per edge (200 boundary points).

**Network and training.** 6 hidden layers × 15 neurons, tanh, Glorot-normal initialization, float64. Adam with a piecewise-constant learning rate (1e-2 until step 1,000; 1e-3 until 3,000; 5e-4 afterwards), 10,000 epochs, about 280 s in the recorded run (hardware not logged). Inputs are scaled to [−1, 1].

## Results

**Training.** Total loss fell from 54.9 at initialization to 0.533 after 10,000 epochs (single run, single seed).

**Comparison with the analytical solution.** Pointwise absolute error maps for uₓ and uᵧ on a 1001 × 1001 evaluation grid are in the notebook. Values below are read from the plot colour bar and are approximate; no scalar error norm has been computed yet.

- Peak amplitude of both displacement components is 1.
- uₓ error is small over most of the plate, with a localized maximum of roughly 0.07 near the top-right corner.
- uᵧ error is larger and spread over the upper half, with a maximum of roughly 0.07 and most of the domain below about 0.04.
- Error is lowest along the Dirichlet edges and highest in the interior and near the loaded edge.

The stress components σₓₓ, σᵧᵧ, σₓᵧ are plotted but not yet compared with the exact stress field.

**Residual diagnostics.** Equilibrium, constitutive, Dirichlet and Neumann residual maps are produced at the end of the notebook.

## Hyperparameter Study

A full grid of 27 configurations was run (layers 6 / 8 / 10, neurons 15 / 20 / 25, dropout 0.1 / 0.2 / 0.3), 1,000 epochs each, with an 80/20 split of collocation points. The lowest validation loss, 2.22, belongs to 6 × 15 with dropout 0.3, which is the configuration used for the 10,000-epoch run.

This ranking should be treated as provisional, for three reasons visible in the notebook:

1. `compute_loss` evaluates the equilibrium and constitutive terms on the global `Residual_train` array and ignores its collocation-point argument. The reported validation loss therefore validates only the boundary terms on held-out points. The interior terms are evaluated on training points.
2. The model is called without `training=True` inside the custom loop, so dropout is most likely inactive in training and validation alike. If so, configurations that differ only in dropout rate are the same model with different random initializations, and the spread in validation loss (about 2.2 to 3.7 for 6 × 15) reflects seed variance rather than a dropout effect. Not yet verified by an ablation.
3. Each configuration was trained once, with no repeated seeds.

## Limitations

- Manufactured, non-dimensional benchmark: no physical material, geometry or load case.
- Accuracy is reported as plots only. No relative L2 error, no stress comparison, no repeated runs.
- Optimizer is Adam only; no L-BFGS refinement stage.
- Loss terms are unweighted; no adaptive weighting.
- The 27-run sweep carries the validation and dropout caveats above.
- A comparison against an FEM solution has not been done. The analytical solution is the only reference.

## Next Steps

- Compute relative L2 error for uₓ, uᵧ, σₓₓ, σᵧᵧ, σₓᵧ against the exact fields and report mean ± spread over several seeds.
- Fix the validation function to use its argument, pass `training=True` where dropout is intended (or drop dropout), and repeat the sweep with multiple seeds.
- Add an L-BFGS stage after Adam and test loss weighting.
- Move from the manufactured field to a case with no closed form (e.g. plate with a hole) and benchmark against FEM.

## Repository Structure

```
PINN_2D_Elasticity/
├── src/        # Core PINN implementation
├── solve/      # Forward pass and loss computation
├── optimize/   # Hyperparameter sweep (27 configs)
├── data/       # LHS collocation point datasets
├── contours/   # Predicted field contour plots
├── Notebook/   # Exploration and visualization notebooks
├── utils/      # Shared utilities
├── env/        # Environment configuration
├── Body Force derivation.docx
├── Neumann_Boundary_Condition_Derivation.pdf
└── solidmechanics_model_stack.sav   # Saved model
```

## Tech Stack

TensorFlow (custom `GradientTape` training loop), NumPy, pyDOE2 (Latin Hypercube Sampling), scikit-learn (train/validation split), Matplotlib.

---

[Portfolio](https://submerged-in-matrix.github.io/projects/pinn-elasticity/) · [GitHub](https://github.com/submerged-in-matrix)
