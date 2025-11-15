from env.module import *
from utils.f_d_exact import *
from src.pred_disp_field import *

# calculate exact solutions
ux_ext = u_x_exact(X.flatten(),Y.flatten())
uy_ext = u_y_exact(X.flatten(),Y.flatten())

# Reshape upred
Ux_ext = ux_ext.numpy().reshape(N+1,N+1)
Uy_ext = uy_ext.numpy().reshape(N+1,N+1)

# fig.tight_layout()
# plt.show()
error_total = [abs(Ux-Ux_ext), abs(Uy-Uy_ext)]
error_total_name = ['point wise error Ux', 'point wise error Uy']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, error_total[i], cmap='jet')
    ax.set_title(error_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle('Point-wise error of PINN solution', fontsize=16)
plt.show()