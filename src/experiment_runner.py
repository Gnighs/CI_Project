import numpy as np
import time
from optimizers import MyGA, CMAES, LBFGSB
from src.simple_mlp import SimpleMLP


class ExperimentRunner:
    def __init__(self, n_trials=30, hidden_sizes=None, random_seed=42):
        if hidden_sizes is None:
            hidden_sizes = [5, 10, 15, 20]
        self.n_trials = n_trials
        self.hidden_sizes = hidden_sizes
        self.random_seed = random_seed
        
    def run_single_trial(self, method_name, X_train, y_train, X_val, y_val, X_test, y_test, n_hidden, trial_idx):
        
        np.random.seed(self.random_seed + trial_idx)
        mlp = SimpleMLP(n_in=X_train.shape[1], n_hidden=n_hidden, n_out=1)
        
        start_time = time.time()
        
        if method_name == "GA":
            opt = MyGA(mlp, X_train, y_train, pop_size=30, n_generations=100, selection_method='tournament')
        elif method_name == "CMAES":
            opt = CMAES(mlp, X_train, y_train, pop_size=30, n_generations=100)
        elif method_name == "LBFGSB":
            opt = LBFGSB(mlp, X_train, y_train, maxiter=500)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        best_genome, history = opt.run()

        elapsed_time = time.time() - start_time

        val_mse = mlp.calculate_mse(X_val, y_val, best_genome)
        test_mse = mlp.calculate_mse(X_test, y_test, best_genome)
        
        return {
            'method': method_name,
            'trial': trial_idx,
            'n_hidden': n_hidden,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'time': elapsed_time,
            'history': history
        }
    
    def run_experiment(self, X_train, y_train, X_val, y_val, X_test, y_test):
        results = []
        methods = ["GA", "CMAES", "LBFGSB"]
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Running {method}")
            print(f"{'='*60}")
            
            for trial in range(self.n_trials):
                print(f"\nTrial {trial + 1}/{self.n_trials}")
                
                trial_results = []
                for n_hidden in self.hidden_sizes:
                    print(f"  Hidden neurons: {n_hidden}")
                    result = self.run_single_trial(
                        method, X_train, y_train, X_val, y_val, X_test, y_test,
                        n_hidden, trial
                    )
                    trial_results.append(result)
                
                best_config = min(trial_results, key=lambda x: x['val_mse'])
                results.append(best_config)
                print(f"  Best: {best_config['n_hidden']} hidden, "
                      f"Val MSE: {best_config['val_mse']:.5f}")
                
                # Store all histories for this trial
                for result in trial_results:
                    if 'history' in result and result['history']:
                        if 'all_histories' not in result:
                            results[-1]['all_histories'] = []
                        results[-1]['all_histories'] = trial_results
        
        return results
    
    def summarize_results(self, results):
        methods = set([r['method'] for r in results])
        summary = {}
        
        for method in methods:
            method_results = [r for r in results if r['method'] == method]
            
            test_mses = [r['test_mse'] for r in method_results]
            val_mses = [r['val_mse'] for r in method_results]
            times = [r['time'] for r in method_results]
            hidden_sizes = [r['n_hidden'] for r in method_results]
            
            summary[method] = {
                'test_mse_mean': np.mean(test_mses),
                'test_mse_std': np.std(test_mses),
                'test_mse_median': np.median(test_mses),
                'val_mse_mean': np.mean(val_mses),
                'val_mse_std': np.std(val_mses),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'hidden_mean': np.mean(hidden_sizes),
                'hidden_mode': max(set(hidden_sizes), key=hidden_sizes.count)
            }
        
        return summary
    
    def run_training_size_experiment(self, train_sizes, X_val, y_val,
                                     X_test, y_test, data_generator):
        results = []
        methods = ["GA", "CMAES", "LBFGSB"]
        
        for train_size in train_sizes:
            print(f"\n{'='*60}")
            print(f"Training size: {train_size}")
            print(f"{'='*60}")
            
            # Generate training data for this size
            X_train, y_train = data_generator.generate_data(train_size)
            
            for method in methods:
                print(f"\nMethod: {method}")
                
                for trial in range(self.n_trials):
                    print(f"  Trial {trial + 1}/{self.n_trials}", end='\r')
                    
                    # Use single hidden size for speed
                    n_hidden = 10
                    result = self.run_single_trial(
                        method, X_train, y_train, X_val, y_val,
                        X_test, y_test, n_hidden, trial
                    )
                    result['train_size'] = train_size
                    results.append(result)
                
                avg_test_mse = np.mean([r['test_mse'] for r in results
                                       if r['method'] == method and
                                       r['train_size'] == train_size])
                print(f"  Avg Test MSE: {avg_test_mse:.5f}                  ")
        
        return results

