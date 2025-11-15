from env.module import *
from utils.material_specs import *

# Define boundary conditions at top 
def Neumann_BC(x, y):
    return (lmda+2*mu)*Q*tf.sin(pi*x)
    # return (lmda+2*mu)*Q*tf.sin(pi*(x-xmin)/(xmax-xmin))