from env.module import *
from utils.data_specs import *
from utils.gradient import get_grad
from data.grid import *
from optimize.data_conv_split import *

def build_and_train_pinn(num_hidden_layers, num_neurons_per_layer, dropout_rate, kernel_initializer, epochs):
    # Build model
    input_layer = tf.keras.Input(shape=(2,))
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    x = scaling_layer(input_layer)
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get('tanh'),
                                 kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    output_Ux = tf.keras.layers.Dense(1)(x)
    output_Uy = tf.keras.layers.Dense(1)(x)
    output_Sxx = tf.keras.layers.Dense(1)(x)
    output_Syy = tf.keras.layers.Dense(1)(x)
    output_Sxy = tf.keras.layers.Dense(1)(x)
    PINN = tf.keras.models.Model(inputs=input_layer, outputs=[output_Ux, output_Uy, output_Sxx, output_Syy, output_Sxy])

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Training loop
    @tf.function
    def train_step():
        loss, grad_theta = get_grad(
            PINN, Residual_train, Boundary_list_train,
            Dirichlet_x_train, Dirichlet_y_train, Neumann_xx_train, Neumann_yy_train)
        optim.apply_gradients(zip(grad_theta, PINN.trainable_variables))
        return loss

    N = epochs
    hist = []
    t0 = time()
    for i in range(N+1):
        loss = train_step()
        if i==0:
            loss0 = loss
        hist.append(loss.numpy()/loss0.numpy())
    # Compute final **validation loss**:
    val_loss, _ = get_grad(
        PINN, Residual_val, Boundary_list_val,
        Dirichlet_x_val, Dirichlet_y_val, Neumann_xx_val, Neumann_yy_val)
    val_loss_value = val_loss.numpy()
    print(f'Finished {N} epochs in {time()-t0:.1f} sec. Final val loss: {val_loss_value:.4e}')
    return val_loss_value