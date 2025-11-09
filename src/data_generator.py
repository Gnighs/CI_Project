import numpy as np

class SyntheticDataGenerator:
    def __init__(self, n_features=5, noise_level=0.1, hardness=1.0, random_seed=42):
        self.n_features = n_features
        self.noise_level = noise_level
        self.hardness = hardness
        self.random_seed = random_seed
        
    def generate_data(self, n_samples):
        np.random.seed(self.random_seed)
        
        X = np.random.randn(n_samples, self.n_features)
        
        # Apply hardness parameter
        # hardness < 1: easier (lower frequency, more linear)
        # hardness = 1: standard difficulty
        # hardness > 1: harder (higher frequency, more nonlinear)
        feature_sum = np.sum(X, axis=1)
        y_clean = np.sin(self.hardness * feature_sum)
        
        # Add non-linear components for higher hardness
        if self.hardness > 1.0:
            y_clean += 0.2 * np.sin(3 * self.hardness * feature_sum)
        
        noise = self.noise_level * np.random.randn(n_samples)
        y = y_clean + noise
        
        return X, y
    
    def generate_train_val_test(self, n_train=200, n_val=40, n_test=1000):
        X_train, y_train = self.generate_data(n_train)
        
        self.random_seed += 1000
        X_val, y_val = self.generate_data(n_val)
        
        self.random_seed += 1000
        X_test, y_test = self.generate_data(n_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def generate_multiple_train_sizes(self, train_sizes, n_val=40, n_test=1000):
        datasets = []
        base_seed = self.random_seed
        
        for n_train in train_sizes:
            self.random_seed = base_seed
            data = self.generate_train_val_test(n_train, n_val, n_test)
            datasets.append((n_train, data))
            base_seed += 10000
        
        return datasets

