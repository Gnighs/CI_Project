from src.optimizer import EvolutionaryAlgorithm
from src.simple_mlp import SimpleMLP, train_with_backprop
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

# Test set for estimating true generalization error
X_test = np.random.randn(1000, n_features)
y_test = np.sin(np.sum(X_test, axis=1)) + 0.1 * np.random.randn(1000)
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

# --- Compare EA with derivative method (scipy.optimize.minimize)
print('\n\n=== Comparison of results ===')

mlp = SimpleMLP(n_in=X_test.shape[1], n_hidden=best_hidden, n_out=1)
test_mse = mlp.calculate_mse(X_test, y_test, best_genome)

mlp_bp = SimpleMLP(n_in=X_train.shape[1], n_hidden=best_hidden, n_out=1)
bp_genome, bp_val_mse = train_with_backprop(mlp_bp, X_train, y_train, X_val, y_val)
bp_test_mse = mlp_bp.calculate_mse(X_test, y_test, bp_genome)

print(f"EA Validation MSE: {best_validation_error:.5f}")
print(f"EA Test MSE: {test_mse:.5f}")
print(f"Derivative-based Validation MSE: {bp_val_mse:.5f}")
print(f"Derivative-based Test MSE: {bp_test_mse:.5f}")