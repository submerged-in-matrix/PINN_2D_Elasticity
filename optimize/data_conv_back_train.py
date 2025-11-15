from env.module import *   
from data.boundary_merged import *
from data.grid import *
from optimize.data_conv_split import *

def to_tensor_safe(arr):
    """Convert arr to tf.Tensor only if it's not already a tensor."""
    if isinstance(arr, tf.Tensor):
        return arr
    return tf.constant(arr, dtype=tf.float64)

def to_tensor_list_safe(arr_list):
    return [to_tensor_safe(arr) for arr in arr_list]

# --- Dirichlet/Neumann BCs (all are now correct shape after your split block!) ---
Dirichlet_x_train = to_tensor_safe(Dirichlet_x_train)
Dirichlet_x_val = to_tensor_safe(Dirichlet_x_val)
Dirichlet_y_train = to_tensor_safe(Dirichlet_y_train)
Dirichlet_y_val = to_tensor_safe(Dirichlet_y_val)
Neumann_xx_train = to_tensor_safe(Neumann_xx_train)
Neumann_xx_val = to_tensor_safe(Neumann_xx_val)
Neumann_yy_train = to_tensor_safe(Neumann_yy_train)
Neumann_yy_val = to_tensor_safe(Neumann_yy_val)

# --- Boundary points (if you use per-side for PINN prediction/BC enforcement) ---
# (You'll need to split/define Boundary_list_train and Boundary_list_val per-side, just like BCs above, if using.)
# Example usage if you have: x_up_train, x_lo_train, x_ri_train, x_le_train
# Boundary_list_train = to_tensor_list_safe([x_up_train, x_lo_train, x_ri_train, x_le_train])
# Boundary_list_val = to_tensor_list_safe([x_up_val, x_lo_val, x_ri_val, x_le_val])

print("Tensors ready! (Shapes):")
print("Residual_train:", Residual_train.shape)
print("Dirichlet_x_train:", Dirichlet_x_train.shape)
print("Dirichlet_y_train:", Dirichlet_y_train.shape)
print("Neumann_xx_train:", Neumann_xx_train.shape)
print("Neumann_yy_train:", Neumann_yy_train.shape)