from env.module import *
from utils.material_specs import *
from utils.data_specs import *
from utils.f_d_exact import *
from utils.neumann import Neumann_BC

# Boundary points #
# Top boundary points
x_up = lhs(1,samples=N_bound,random_state=123)   # Latin Hypercube Sampling (LHS) to get 50 points in 1D (x-direction) between 0 and 1.
x_up = xmin + (xmax-xmin)*x_up                   # Maps these points to the actual x-range of the domain.
y_up = np.empty(len(x_up))[:,None]               # [:, None] reshapes the array to have shape (N_bound, 1).
y_up.fill(ymax)                                  # Creates an array of the same length as x_up, filled with the maximum y-value (ymax).

# Initialize Placeholder for Boundary Datas
disp_up = np.empty([len(x_up),2])                # a (50, 2) array meant to store displacement data at the top edge (for u_x and u_y).

# Fill the boundary data with the values of the external functions u_x and u_y at the boundary points.
disp_up[:,0,None] = u_x_exact(x_up, y_up)
disp_up[:,1,None] = u_y_exact(x_up, y_up)
x_up_train = np.hstack((x_up, y_up))             # as input to the PINN for training the boundary conditions.
 
Dirichlet_up_train = np.zeros([len(x_up),1])     # Initialize Dirichlet boundary condition at the top boundary to zero
Neumann_up_train = Neumann_BC(x_up, y_up)        # computes the Neumann BC (traction) in the y-direction at the top boundary. bottom is fixed.

x_low = lhs(1,samples=N_bound,random_state=123)
x_low = xmin + (xmax-xmin)*x_low
y_low = np.empty(len(x_low))[:,None]
y_low.fill(ymin)                                  # Creates an array of the same length as x_lo, filled with the minimum y-value (ymin).

# Initialize Placeholder for Boundary Data
disp_low = np.empty([len(x_low),2])
disp_low[:,0, None] = u_x_exact(x_low, y_low)
disp_low[:,1, None] = u_y_exact(x_low, y_low)
x_low_train = np.hstack((x_low, y_low))

# Initialize Dirichlet boundary condition at the bottom boundary to zero and Neumann boundary condition does not apply at the bottom boundary.
Dirichlet_x_low_train = np.zeros([len(x_low),1])
Dirichlet_y_low_train = np.zeros([len(x_low),1])

print(disp_up.shape)
print(disp_low.shape)