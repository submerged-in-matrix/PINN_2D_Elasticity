from env.module import *
from utils.dirich_residual import *
from src.network import PINN
from data.boundary_merged import *

# --- 1. Compute Dirichlet Residuals ---
r_ux, r_uy = get_Dirichlet_r(PINN, Boundary_list, Dirichlet_x_train, Dirichlet_y_train)

# --- 2. Concatenate Boundary Coordinates ---

x_up_train, x_lo_train, x_ri_train, x_le_train = Boundary_list

# For r_ux: top and bottom boundaries
X_bc_x = np.concatenate([x_up_train, x_lo_train], axis=0)  # shape (N1+N2, 2)

# For r_uy: bottom, right, left boundaries
X_bc_y = np.concatenate([x_lo_train, x_ri_train, x_le_train], axis=0)  # shape (N3+N4+N5, 2)

# --- 3. Plot Dirichlet Residuals ---
import matplotlib.pyplot as plt

# Flatten if needed
r_ux_plot = r_ux.numpy().flatten()
r_uy_plot = r_uy.numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# X-displacement residuals (Dirichlet)
c1 = axes[0].scatter(X_bc_x[:,0], X_bc_x[:,1], c=r_ux_plot, cmap='coolwarm', s=25)
fig.colorbar(c1, ax=axes[0])
axes[0].set_title(r'Dirichlet Residual $r^D_x$ (fixed $u_x$)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Y-displacement residuals (Dirichlet)
c2 = axes[1].scatter(X_bc_y[:,0], X_bc_y[:,1], c=r_uy_plot, cmap='coolwarm', s=25)
fig.colorbar(c2, ax=axes[1])
axes[1].set_title(r'Dirichlet Residual $r^D_y$ (fixed $u_y$)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()