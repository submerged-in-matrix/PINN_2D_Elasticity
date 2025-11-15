from env.module import *
from src.network import *
from src.train import N

print("----- PINN Model Hyperparameters -----")
print(f"Hidden layers: {num_hidden_layers}")
print(f"Neurons per layer: {num_neurons_per_layer}")
print(f"Activation function: tanh")
print(f"Dropout rate: 0.2")
print(f"Kernel initializer: glorot_normal")
print(f"Learning rate schedule: 0-1000 steps: 0.01, 1000-3000: 0.001, 3000+: 0.0005")
print(f"Optimizer: Adam")
print(f"Epochs: {N}")
print("---------------------------------------")