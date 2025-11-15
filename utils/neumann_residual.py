from env.module import *

# boundary condition with fixed stress (Neumann boundary condition)
def Neumann_r_xx(Sxx_ri, Sxx_le, Sxx_b_train):
    return tf.concat([Sxx_ri,Sxx_le], 0)-Sxx_b_train

def Neumann_r_yy(Syy_up, Syy_b_train):
    return Syy_up-Syy_b_train

def get_Neumann_r(model, Boundary_list , Sxx_b_train, Syy_b_train):
    x_up_train, x_lo_train, x_ri_train, x_le_train = Boundary_list[0], \
                    Boundary_list[1], Boundary_list[2], Boundary_list[3]
    # A tf.GradientTape is used to compute derivatives in TensorFlow

    ## up boundary
    x_up = tf.constant(x_up_train[:, 0:1])
    y_up = tf.constant(x_up_train[:, 1:2])

    Ux_up, Uy_up, Sxx_up, Syy_up, Sxy_up  = model(tf.stack([x_up[:,0], y_up[:,0]], axis=1))

    ## right boundary
    x_ri = tf.constant(x_ri_train[:, 0:1])
    y_ri = tf.constant(x_ri_train[:, 1:2])

    Ux_ri, Uy_ri, Sxx_ri, Syy_ri, Sxy_ri = model(tf.stack([x_ri[:,0], y_ri[:,0]], axis=1))

    ## left boundary    
    x_le = tf.constant(x_le_train[:, 0:1])
    y_le = tf.constant(x_le_train[:, 1:2])

    Ux_le, Uy_le, Sxx_le, Syy_le, Sxy_le = model(tf.stack([x_le[:,0], y_le[:,0]], axis=1))
    
    return Neumann_r_xx(Sxx_ri, Sxx_le, Sxx_b_train), Neumann_r_yy(Syy_up, Syy_b_train)