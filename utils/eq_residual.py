from env.module import *
from utils.f_d_exact import *

# Define residual of the PDE in x direction                                 # Commented out  fun_r_x, fun_r_y: starting from strain-displacement relations rather than stress-divergence relations.
def eq_r_x(x, y, dsxxdx, dsxydy):                                           # dsxxdx:rate of change of the normal stress in the x direction with respect to x.
    # return (lmda+2*mu)*ux_xx+lmda*uy_xy+mu*(ux_yy+uy_xy)-f_x_ext(x,y)     # dsxydy: rate of change of the shear stress in the x-y plane with respect to y.
    return dsxxdx+dsxydy-f_x_exact(x,y)
# Define residual of the PDE in y direction
def eq_r_y(x, y, dsxydx, dsyydy):                                           # dsxydx: rate of change of the shear stress in the x-y plane with respect to x.
    # return mu*(ux_xy+uy_xx)+(lmda+2*mu)*uy_yy+lmda*ux_xy-f_y_ext(x,y)     # dsyydy: rate of change of the normal stress in the y direction with respect to y.
    return dsxydx+dsyydy-f_y_exact(x,y)
    

def get_eq_r(model, Residual_train):
    # Split x and y to compute partial derivatives
    x = tf.constant(Residual_train[:, 0:1])
    y = tf.constant(Residual_train[:, 1:2])
    # x = Residual_train[:, 0:1]
    # y = Residual_train[:, 1:2]
    # A tf.GradientTape is used to compute derivatives in TensorFlow

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        # Determine residual 
        Ux, Uy, Sxx, Syy, Sxy = model(tf.stack([x[:,0], y[:,0]], axis=1))
        sxx = Sxx
        syy = Syy
        sxy = Sxy
        # Compute gradient u_x within the GradientTape, since we need second derivatives.
    dsxxdx, dsxxdy = tape2.gradient(sxx, (x,y))
    dsyydx, dsyydy = tape2.gradient(syy, (x,y))
    dsxydx, dsxydy = tape2.gradient(sxy, (x,y))

    del tape2
    return eq_r_x(x, y, dsxxdx, dsxydy), eq_r_y(x, y, dsxydx, dsyydy)