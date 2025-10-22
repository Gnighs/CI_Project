from scipy.optimize import minimize
import numpy as np

def train_with_backprop(mlp, X_train, y_train, X_val, y_val):
    initial_genome = np.random.randn(mlp.n_params())

    def objective(genome):
        return mlp.calculate_mse(X_train, y_train, genome)

    result = minimize(objective, initial_genome, method='L-BFGS-B', options={'maxiter':500})
    best_genome = result.x

    mse = mlp.calculate_mse(X_val, y_val, best_genome)
    return best_genome, mse

class SimpleMLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

    def n_params(self):
        return (self.n_in * self.n_hidden) + self.n_hidden + (self.n_hidden * self.n_out) + self.n_out

    # Assume genome represents weights in a flattened vector
    def _decode_weights(self, genome):
        if len(genome) != self.n_params():
            raise ValueError(f"Genome is not the right size.")
        
        W1 = genome[0: self.n_in * self.n_hidden].reshape(self.n_in, self.n_hidden)
        idx = self.n_in * self.n_hidden

        b1 = genome[idx: idx + self.n_hidden]
        idx += self.n_hidden

        W2 = genome[idx: idx + self.n_hidden * self.n_out].reshape(self.n_hidden, self.n_out)
        idx += self.n_hidden * self.n_out

        b2 = genome[idx: idx + self.n_out]
        return W1, b1, W2, b2

    def _forward(self, X, genome):
        W1, b1, W2, b2 = self._decode_weights(genome)
        
        hidden_values = X.dot(W1) + b1
        hidden_values = np.maximum(0, hidden_values) # ReLU
        output_values = hidden_values.dot(W2) + b2

        return output_values

    def calculate_mse(self, X, y, genome):
        y_pred = self._forward(X, genome)
        mse = np.mean((y_pred - y)**2)

        return mse