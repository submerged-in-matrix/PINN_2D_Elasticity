from env.module import *
from src.network import PINN
from src.pred_disp_field import *

# Determine predictions of stress tensor components
# Note: The stress tensor components are derived from the displacements using the constitutive relations.
Ux_pred, Uy_pred, Sxx_pred, Syy_pred, Sxy_pred = PINN(tf.cast(Xgrid,DTYPE))
sxx_pred = Sxx_pred
syy_pred = Syy_pred
sxy_pred = Sxy_pred

# Reshape upred
Sxx = sxx_pred.numpy().reshape(N+1,N+1)
Syy = syy_pred.numpy().reshape(N+1,N+1)
Sxy = sxy_pred.numpy().reshape(N+1,N+1)

S_total = [Sxx, Syy, Sxy]
S_total_name = ['Sxx_NN', 'Syy_NN', 'Sxy_NN']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
for i,ax in enumerate(axes.flat):
    im = ax.pcolor(X, Y, S_total[i], cmap='seismic', vmin=-10, vmax=10)
    ax.set_title(S_total_name[i])

fig.colorbar(im, ax=axes.ravel().tolist())
plt.suptitle('Stress tensor components by PINN', fontsize=16)
plt.show()

#plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300);