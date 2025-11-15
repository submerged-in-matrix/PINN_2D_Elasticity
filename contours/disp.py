import matplotlib.pyplot as plt
from src.pred_disp_field import Ux_pred, Uy_pred, Xgrid

# Ux displacement
plt.figure(figsize=(6,5))
plt.tricontourf(Xgrid[:,0], Xgrid[:,1], Ux_pred.numpy().flatten(), 20, cmap='viridis')
plt.colorbar(label='$u_x$ displacement')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted $u_x$ Field')
plt.tight_layout()
plt.show()

# Uy displacement
plt.figure(figsize=(6,5))
plt.tricontourf(Xgrid[:,0], Xgrid[:,1], Uy_pred.numpy().flatten(), 20, cmap='plasma')
plt.colorbar(label='$u_y$ displacement')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted $u_y$ Field')
plt.tight_layout()
plt.show()