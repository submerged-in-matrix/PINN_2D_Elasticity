from env.module import *    

# boundary condition with fixed displacement (Dirichlet boundary condition)
def Dirichlet_r_x(ux_up, ux_lo, ux_b_train):                                     # Calculates the residuals for the x-displacement at the top and bottom boundaries of the domain.
    # ux_b_temp = tf.stack([ux_up, ux_lo],axis=0)
    return   tf.concat([ux_up, ux_lo], 0)-ux_b_train

def Dirichlet_r_y(uy_lo, uy_ri,uy_le, uy_b_train):
    # uy_b_temp = tf.stack([uy_lo, uy_ri, uy_le],axis=0)
    return  tf.concat([uy_lo, uy_ri, uy_le], 0)-uy_b_train

def get_Dirichlet_r(model, Boundary_list, ux_b_train, uy_b_train):
    x_up_train, x_lo_train, x_ri_train, x_le_train = Boundary_list[0], \
                    Boundary_list[1], Boundary_list[2], Boundary_list[3]
    # Split t and x to compute partial derivatives
    x_up, y_up = x_up_train[:, 0:1], x_up_train[:,1:2]
    x_lo, y_lo = x_lo_train[:, 0:1], x_lo_train[:,1:2]
    x_ri, y_ri = x_ri_train[:, 0:1], x_ri_train[:,1:2]
    x_le, y_le = x_le_train[:, 0:1], x_le_train[:,1:2]
    # Determine residual 
    Ux_up, Uy_up, Sxx, Syy, Sxy  = model(tf.stack([x_up[:,0], y_up[:,0]], axis=1))
    ux_up, _ = Ux_up, Uy_up
    Ux_lo, Uy_lo, Sxx, Syy, Sxy  = model(tf.stack([x_lo[:,0], y_lo[:,0]], axis=1))
    ux_lo, uy_lo = Ux_lo, Uy_lo
    Ux_ri, Uy_ri, Sxx, Syy, Sxy  = model(tf.stack([x_ri[:,0], y_ri[:,0]], axis=1))
    _, uy_ri = Ux_ri, Uy_ri
    Ux_le, Uy_le, Sxx, Syy, Sxy  = model(tf.stack([x_le[:,0], y_le[:,0]], axis=1))
    _, uy_le = Ux_le, Uy_le
    
    return Dirichlet_r_x(ux_up, ux_lo, ux_b_train), Dirichlet_r_y(uy_lo, uy_ri,uy_le, uy_b_train)