from env.module import *
from utils.eq_residual import *
from utils.consti_residual import * 
from utils.dirich_residual import *
from utils.neumann_residual import *
from data.grid import Residual_train

def compute_loss(model, X_col_train, X_train_list, ux_b_train, uy_b_train, Sxx_b_train, Syy_b_train):
    x_up_train, x_lo_train, x_ri_train, x_le_train = X_train_list[0],\
                    X_train_list[1], X_train_list[2], X_train_list[3]
                    
    # Compute phi^r, use absolute error
    rx, ry = get_eq_r(model, Residual_train)
    # phi_r = tf.reduce_mean(tf.square(rx)) + tf.reduce_mean(tf.square(ry))     # MSE loss
    phi_r = tf.reduce_mean(tf.abs(rx)) + tf.reduce_mean(tf.abs(ry))             # MAE loss

    rx_const, ry_const, rxy_const = get_consti_r(model, Residual_train)
    # phi_r_const = tf.reduce_mean(tf.square(rx_const)) + tf.reduce_mean(tf.square(ry_const))+tf.reduce_mean(tf.square(rxy_const))
    phi_r_const = tf.reduce_mean(tf.abs(rx_const)) + tf.reduce_mean(tf.abs(ry_const))+tf.reduce_mean(tf.abs(rxy_const))

    # Compute phi^b
    r_ux, r_uy = get_Dirichlet_r(model, X_train_list, ux_b_train, uy_b_train)
    # phi_r_u = tf.reduce_mean(tf.square(r_ux)) + tf.reduce_mean(tf.square(r_uy))
    phi_r_u = tf.reduce_mean(tf.abs(r_ux)) + tf.reduce_mean(tf.abs(r_uy))
    
    # Compute phi^b
    r_Sxx, r_Syy = get_Neumann_r(model, X_train_list, Sxx_b_train, Syy_b_train)
    # phi_r_S = tf.reduce_mean(tf.square(r_Sxx)) + tf.reduce_mean(tf.square(r_Syy))
    phi_r_S = tf.reduce_mean(tf.abs(r_Sxx)) + tf.reduce_mean(tf.abs(r_Syy))
    
    loss = phi_r+ phi_r_const + phi_r_u + phi_r_S 

    return loss