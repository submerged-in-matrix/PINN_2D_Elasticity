from env.module import *
from utils.data_specs import *

def initial_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    #  Input Layer
    model.add(tf.keras.Input(shape=(2,)))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(
                lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Append hidden layers. 
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),               # activation is tanh, since the outputs were rescaled to [-1, 1] range.
            kernel_initializer='glorot_normal'))                       # A balanced way to start weights, avoiding exploding/vanishing gradients.
        model.add(tf.keras.layers.Dropout(0.2))                        # Randomly drops neurons during training to reduce overfitting.
    # Output is two-dimensional
    model.add(tf.keras.layers.Dense(2))
    
    return model

num_hidden_layers, num_neurons_per_layer = 6, 15         ## Started with 8 hidden layers and 20 neurons per layer, but reduced to 6 layers, 15 neurons  for efficient performance.
input_layer = tf.keras.Input(shape=(2,))

# Introduce a scaling layer to map input to [lb, ub]
scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)

x = scaling_layer(input_layer)

# Append hidden layers
for _ in range(num_hidden_layers):
    x = tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get('tanh'),\
        kernel_initializer='glorot_normal')(x)                                                          # Randomly drops neurons during training to reduce overfitting.
    x = tf.keras.layers.Dropout(0.3)(x)    # started with 0.2 dropout, but increased to 0.3 for better regularization.
    
# Output is two-dimensional
output_Ux = tf.keras.layers.Dense(1)(x)
output_Uy = tf.keras.layers.Dense(1)(x)
output_Sxx = tf.keras.layers.Dense(1)(x)
output_Syy = tf.keras.layers.Dense(1)(x)
output_Sxy = tf.keras.layers.Dense(1)(x)

PINN = tf.keras.models.Model(inputs=input_layer, outputs=[output_Ux, output_Uy, output_Sxx, output_Syy, output_Sxy])

# import pickle
# filename = 'solidmechanics_model_stack.sav'
# pickle.dump(PINN, open(filename, 'wb'))

PINN.summary()