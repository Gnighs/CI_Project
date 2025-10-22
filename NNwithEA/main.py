from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from src.MLP import SimpleMLP
import numpy as np

n_samples = 200
n_features = 5

# Dummy dataset
X_train = np.random.randn(n_samples, n_features)
y_train = np.sin(np.sum(X_train, axis=1)) + 0.1 * np.random.randn(n_samples)

mlp = SimpleMLP(n_in=X_train.shape[1], n_hidden=10, n_out=1)
ea = EvolutionaryAlgorithm(mlp, X_train, y_train, pop_size=20, n_generations=50)

best_genome = ea.run()

# Evaluate on training set
mse = mlp.calculate_mse(X_train, y_train, best_genome)
print("Best MSE:", mse)