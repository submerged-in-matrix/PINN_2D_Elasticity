from env.module import *
from utils.material_specs import *
from utils.data_specs import *
from utils.f_d_exact import *
from utils.neumann import Neumann_BC

y_right = lhs(1,samples=N_bound,random_state=123)
y_right = ymin + (ymax-ymin)*y_right
x_right = np.empty(len(y_right))[:,None]
x_right.fill(xmax)

disp_right = np.empty([len(x_right),2])
disp_right[:,0, None] = u_x_exact(x_right, y_right)
disp_right[:,1, None] = u_y_exact(x_right, y_right)

x_right_train = np.hstack((x_right, y_right)) 
Dirichlet_right_train = np.zeros([len(x_right),1])                  # meaning no vertical movement. (dirichlet condition)
Neumann_right_train = np.zeros([len(x_right),1])                    # traction-free or stress-free condition (Neumann BC)

y_left = lhs(1,samples=N_bound,random_state=123)
y_left = ymin + (ymax-ymin)*y_left
x_left = np.empty(len(y_left))[:,None]
x_left.fill(xmin)

disp_left = np.empty([len(x_left),2])
disp_left[:,0, None] = u_x_exact(x_left, y_left)
disp_left[:,1, None] = u_y_exact(x_left, y_left)

x_left_train = np.hstack((x_left, y_left))
Dirichlet_left_train = np.zeros([len(x_left),1])
Neumann_left_train = np.zeros([len(x_left),1])

print(disp_right.shape)
print(disp_left.shape)