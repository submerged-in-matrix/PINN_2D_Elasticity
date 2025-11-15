from env.module import *
from utils.gradient import *
from data.boundary_merged import *
from data.grid import Residual_train
from src.network import *
from optimize.adam_piecewise_decay import optim

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(PINN, Residual_train, Boundary_list, Dirichlet_x_train, Dirichlet_y_train, Neumann_xx_train, Neumann_yy_train)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, PINN.trainable_variables))
    
    return loss

# Number of training epochs
N = 10000
hist = []

# Start timer
t0 = time()

for i in range(N+1):
  loss = train_step()
   
  if i==0:
    loss0 = loss
    # Append current loss to hist
  hist.append(loss.numpy()/loss0.numpy())
    
  # Output current loss after some iterations
  if i%1000 == 0:
      print('It {:05d}: loss = {:10.8e}'.format(i,loss))
        
# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))