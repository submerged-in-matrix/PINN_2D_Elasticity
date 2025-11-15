from env.module import *

# We choose a piecewise decay of the learning rate, i.e., the step size in the gradient descent type algorithm, the first 1000 steps use a learning rate of 0.01  
# from 1000 - 3000: learning rate = 0.001 from 3000 onwards: learning rate = 0.0005

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])

# Choose the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr)