import matplotlib.pyplot as plt
from src.pred_disp_field import Ux_pred, Uy_pred, Xgrid
from src.pred_stress_field import Sxx_pred, Syy_pred, Sxy_pred

# Sxx stress
plt.figure(figsize=(6,5))
plt.tricontourf(Xgrid[:,0], Xgrid[:,1], Sxx_pred.numpy().flatten(), 20, cmap='coolwarm')
plt.colorbar(label='$\sigma_{xx}$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted $\sigma_{xx}$ Field')
plt.tight_layout()
plt.show()

# Syy stress
plt.figure(figsize=(6,5))
plt.tricontourf(Xgrid[:,0], Xgrid[:,1], Syy_pred.numpy().flatten(), 20, cmap='coolwarm')
plt.colorbar(label='$\sigma_{yy}$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted $\sigma_{yy}$ Field')
plt.tight_layout()
plt.show()

# Sxy stress
plt.figure(figsize=(6,5))
plt.tricontourf(Xgrid[:,0], Xgrid[:,1], Sxy_pred.numpy().flatten(), 20, cmap='coolwarm')
plt.colorbar(label='$\sigma_{xy}$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted $\sigma_{xy}$ Field')
plt.tight_layout()
plt.show()
