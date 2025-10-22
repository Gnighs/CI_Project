from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from src.MLP import SimpleMLP
import numpy as np

n_samples = 200
n_features = 5

# --- Dummy dataset ---
X = np.random.randn(n_samples, n_features)
y = np.sin(np.sum(X, axis=1)) + 0.1 * np.random.randn(n_samples)

split_ratio = 0.8
split_idx = int(n_samples * split_ratio)

X_train = X[:split_idx]
y_train = y[:split_idx]

X_val = X[split_idx:]
y_val = y[split_idx:]
# --- ---

hidden_sizes = [5, 10, 15]
best_validation_error = float("inf")
best_genome = None
best_hidden = None

for n_hidden in hidden_sizes:
    mlp = SimpleMLP(n_in=X_train.shape[1], n_hidden=n_hidden, n_out=1)
    
    ea = EvolutionaryAlgorithm(mlp, X_train, y_train, pop_size=20, n_generations=50)
    genome = ea.run()

    mse = mlp.calculate_mse(X_val, y_val, genome)
    
    print(f"Hidden neurons: {n_hidden}, Validation MSE: {mse:.5f}")
    
    if mse < best_validation_error:
        best_validation_error = mse
        best_genome = genome
        best_hidden = n_hidden

print(f"Best hidden size: {best_hidden}, Best validation MSE: {best_validation_error:.5f}")
