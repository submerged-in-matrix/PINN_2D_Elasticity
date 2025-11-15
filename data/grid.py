from env.module import *
from utils.data_specs import *
from utils.material_specs import *
# Generate residual points in the domain using Latin Hypercube Sampling
grid_pt = lhs(2, N_r)

# Rescale the grid points to fit within the defined domain
grid_pt[:,0] = xmin + (xmax-xmin)*grid_pt[:,0]
grid_pt[:,1] = ymin + (ymax-ymin)*grid_pt[:,1]

# Separate x and y Coordinates
x_displacement_grid = grid_pt[:,0]
y_displacement_grid = grid_pt[:,1]

## compute the body forces at each collocation point using the force functions defined earlier.
# ff_x = np.asarray([ f_x_ext(xf[j],yf[j]) for j in range(len(yf))])
# ff_y = np.asarray([ f_y_ext(xf[j],yf[j]) for j in range(len(yf))])
# f_x_train = ff_x
# f_y_train = ff_y

# Combine Coordinates
Residual_train = np.hstack((x_displacement_grid[:,None], y_displacement_grid[:,None]))

print("The residual training data points:", Residual_train.shape)   