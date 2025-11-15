from env.module import *
from src.network import PINN
from utils.adaptive_model import *

# Define your search space
search_space = {
    'num_hidden_layers': [6, 8, 10],
    'num_neurons_per_layer': [15, 20, 25],
    'dropout_rate': [0.1, 0.2, 0.3],
    'kernel_initializer': ['glorot_normal'],
    'epochs': [1000],
}

# Create a list of all combinations
keys = list(search_space.keys())
combinations = list(product(*[search_space[k] for k in keys]))

# Store results
results = []
param_list = []

# Evaluate all combinations
for params in combinations:
    config = dict(zip(keys, params))
    print(f"Trying config: {config}")
    val_loss = build_and_train_pinn(**config)
    results.append(val_loss)
    param_list.append(config)

# Find the best result
best_index = results.index(min(results))
best_config = param_list[best_index]

print("\nBest score (validation loss):", results[best_index])
print("Best parameters:")
for k, v in best_config.items():
    print(f"{k}: {v}")

# Plot convergence
plt.plot(results)
plt.xlabel('Trial')
plt.ylabel('Validation Loss')
plt.title('Search Progress')
plt.show()