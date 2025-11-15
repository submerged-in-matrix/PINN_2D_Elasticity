from env.module import *
from data.boundary_UL import *
from data.boundary_RL import *

# Combine all boundary data into a single training set
Boundary_train = np.concatenate((x_up_train, x_low_train, x_right_train, x_left_train))
Boundary_list = [x_up_train, x_low_train, x_right_train, x_left_train]

Dirichlet_x_train = np.concatenate((Dirichlet_up_train, Dirichlet_x_low_train))
Dirichlet_y_train = np.concatenate((Dirichlet_y_low_train, Dirichlet_right_train, Dirichlet_left_train))
Neumann_xx_train = np.concatenate((Neumann_right_train, Neumann_left_train))
Neumann_yy_train = Neumann_up_train

print( "The boundary data points, Dirichlet boundary data points (for x & y), Neumann boundary data points (for x & y):",  Boundary_train.shape, Dirichlet_x_train.shape, Dirichlet_y_train.shape, 
      Neumann_xx_train.shape, Neumann_yy_train.shape)