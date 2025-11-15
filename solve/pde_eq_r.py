import matplotlib.pyplot as plt
from src.pred_disp_field import Xgrid
from src.network import PINN
from utils.eq_residual import get_eq_r

# Compute equilibrium residuals across the grid
rx_grid, ry_grid = get_eq_r(PINN, Xgrid)

# Convert to numpy arrays (if not already)
rx_plot = rx_grid.numpy().flatten()
ry_plot = ry_grid.numpy().flatten()

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

c1 = axes[0].tricontourf(Xgrid[:,0], Xgrid[:,1], rx_plot, 20, cmap='RdBu_r')
fig.colorbar(c1, ax=axes[0])
axes[0].set_title(r'PDE Residual $r_x$ (x-direction)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

c2 = axes[1].tricontourf(Xgrid[:,0], Xgrid[:,1], ry_plot, 20, cmap='RdBu_r')
fig.colorbar(c2, ax=axes[1])
axes[1].set_title(r'PDE Residual $r_y$ (y-direction)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()