from env.module import *
from utils.consti_residual import get_consti_r
from src.network import PINN
from data.grid import Residual_train

# Compute constitutive residuals at collocation points
rx_const, ry_const, rxy_const = get_consti_r(PINN, Residual_train)

# Flatten arrays for plotting
rx_const_plot = rx_const.numpy().flatten()
ry_const_plot = ry_const.numpy().flatten()
rxy_const_plot = rxy_const.numpy().flatten()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

c1 = axes[0].tricontourf(Residual_train[:,0], Residual_train[:,1], rx_const_plot, 20, cmap='RdBu_r')
fig.colorbar(c1, ax=axes[0])
axes[0].set_title(r'Constitutive Residual $r^{const}_x$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

c2 = axes[1].tricontourf(Residual_train[:,0], Residual_train[:,1], ry_const_plot, 20, cmap='RdBu_r')
fig.colorbar(c2, ax=axes[1])
axes[1].set_title(r'Constitutive Residual $r^{const}_y$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

c3 = axes[2].tricontourf(Residual_train[:,0], Residual_train[:,1], rxy_const_plot, 20, cmap='RdBu_r')
fig.colorbar(c3, ax=axes[2])
axes[2].set_title(r'Constitutive Residual $r^{const}_{xy}$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

plt.tight_layout()
plt.show()