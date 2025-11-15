from env.module import *
from utils.material_specs import *

# Define residual of the PDE in x direction
def consti_r_x(x, y, duxdx, duydy,Sxx):
    # return (lmda+2*mu)*ux_xx+lmda*uy_xy+mu*(ux_yy+uy_xy)-f_x_ext(x,y)
    return (lmda+2*mu)*duxdx+lmda*duydy-Sxx
# Define residual of the PDE in y direction
def consti_r_y(x, y, duxdx, duydy, Syy):
    return (lmda+2*mu)*duydy+lmda*duxdx-Syy
    # return mu*(ux_xy+uy_xx)+(lmda+2*mu)*uy_yy+lmda*ux_xy-f_y_ext(x,y)
# Define residual of the PDE in xy direction
def consti_r_xy(x, y, duxdy, duydx, Sxy):
    return 2*mu*0.5*(duxdy+duydx)-Sxy
    # return mu*(ux_xy+uy_xx)+(lmda+2*mu)*uy_yy+lmda*ux_xy-f_y_ext(x,y)

def get_consti_r(model, Residual_train):
    # Split x and y to compute partial derivatives
    x = tf.constant(Residual_train[:, 0:1])    # Commented out for bayesian optimization
    y = tf.constant(Residual_train[:, 1:2])
    # x = Residual_train[:, 0:1]
    # y = Residual_train[:, 1:2]
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape2:
#             # Variables x and y are watched during tape
#             # to compute derivatives u_t and u_x
        tape2.watch(x)
        tape2.watch(y)
        # Determine residual 
        # U = model(tf.stack([x[:,0], y[:,0]], axis=1))
        # ux = U[:,0]
        # uy = U[:,1]
        Ux, Uy, Sxx, Syy, Sxy = model(tf.stack([x[:,0], y[:,0]], axis=1))
        ux = Ux
        uy = Uy
        # Compute gradient u_x within the GradientTape, since we need second derivatives.
    duxdx, duxdy = tape2.gradient(ux, (x,y))
    duydx, duydy = tape2.gradient(uy, (x,y))
    del tape2
    return consti_r_x(x, y, duxdx, duydy,Sxx), consti_r_y(x, y, duxdx, duydy, Syy), consti_r_xy(x, y, duxdy, duydx, Sxy)