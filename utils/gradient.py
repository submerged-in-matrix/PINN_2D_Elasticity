from env.module import *
from utils.total_residual import *

def get_grad(model, X_col_train, X_train_list, ux_b_train, uy_b_train, Sxx_b_train, Syy_b_train):
    
    with tf.GradientTape(persistent=True) as tape:    # persistent=True allows the tape to compute gradients multiple times if needed 
        
        # This tape is for derivatives with respect to trainable variables
        #tape.watch(model.trainable_variables)
        
        loss = compute_loss(model, X_col_train, X_train_list, ux_b_train, uy_b_train, Sxx_b_train, Syy_b_train)

    g = tape.gradient(loss, model.trainable_variables)
    del tape # Release resources held by the tape

    return loss, g