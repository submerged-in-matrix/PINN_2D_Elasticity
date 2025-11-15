from env.module import *
from utils.neumann_residual import *
from src.network import PINN
from data.boundary_merged import *

# --- 1. Compute Neumann Residuals ---
r_Sxx, r_Syy = get_Neumann_r(PINN, Boundary_list, Neumann_xx_train, Neumann_yy_train)

# --- 2. Prepare Corresponding Boundary Coordinates ---
import numpy as np

x_up_train, x_lo_train, x_ri_train, x_le_train = Boundary_list

# Sxx BC residuals: right and left boundaries
X_bc_Sxx = np.concatenate([x_ri_train, x_le_train], axis=0)

# Syy BC residuals: upper boundary
X_bc_Syy = x_up_train  # already the correct shape

# --- 3. Plot Neumann Residuals ---
import matplotlib.pyplot as plt

r_Sxx_plot = r_Sxx.numpy().flatten()
r_Syy_plot = r_Syy.numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Sxx boundary residuals (Neumann)
c1 = axes[0].scatter(X_bc_Sxx[:,0], X_bc_Sxx[:,1], c=r_Sxx_plot, cmap='coolwarm', s=25)
fig.colorbar(c1, ax=axes[0])
axes[0].set_title(r'Neumann Residual $r^N_{xx}$ (fixed $\sigma_{xx}$)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Syy boundary residuals (Neumann)
c2 = axes[1].scatter(X_bc_Syy[:,0], X_bc_Syy[:,1], c=r_Syy_plot, cmap='coolwarm', s=25)
fig.colorbar(c2, ax=axes[1])
axes[1].set_title(r'Neumann Residual $r^N_{yy}$ (fixed $\sigma_{yy}$)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()