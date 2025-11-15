from env.module import *
from utils.precision import DTYPE
from utils.material_specs import *
# Set number of data points
# N_0 = 50
N_bound = 50            # 50 points on each boundary
N_r = 1000              # 1000 residual points inside the domain

# Lower bounds
lb = tf.constant([xmin, ymin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, ymax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)