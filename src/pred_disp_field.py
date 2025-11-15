from env.module import *
from utils.data_specs import *
from src.network import PINN

# Set up meshgrid
N = 1000
xspace = np.linspace(lb[0], ub[0], N + 1)
yspace = np.linspace(lb[1], ub[1], N + 1)
X, Y = np.meshgrid(xspace, yspace)
Xgrid = np.vstack([X.flatten(),Y.flatten()]).T # 2×(N+1)² matrix

# Determine predictions of u(t, x)
Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = PINN(tf.cast(Xgrid,DTYPE))
ux_pred = Ux_pred
uy_pred = Uy_pred 

# Reshape upred
Ux = ux_pred.numpy().reshape(N+1,N+1)
Uy = uy_pred.numpy().reshape(N+1,N+1)

U_total = [Ux, Uy]
U_total_name = ['Ux_NN', 'Uy_NN']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, U_total[i], cmap='seismic', vmin=-0.8, vmax=0.8)
    ax.set_title(U_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle('Normal Displacement field predicted by PINN', fontsize=12)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

#plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300);