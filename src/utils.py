from src.statistical_analysis import StatisticalAnalyzer
from src.data_generator import SyntheticDataGenerator
from src.experiment_runner import ExperimentRunner
from src.visualization import Visualizer
import json
import os


class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        mode = self.config['mode']
        print(f"\nRunning experiment mode: {mode}")
        data_gen = SyntheticDataGenerator(
            n_features=self.config['n_features'],
            noise_level=self.config['noise_level'],
            hardness=self.config.get('hardness', 1.0),
            random_seed=self.config['random_seed']
        )

        if mode == 'train_size':
            return self._run_train_size_experiment(data_gen)
        else:
            return self._run_standard_experiment(data_gen)

    def _run_train_size_experiment(self, data_gen):
        train_sizes = self.config['train_sizes']
        n_val, n_test = self.config['n_val'], self.config['n_test']
        _, (X_val, y_val), (X_test, y_test) = data_gen.generate_train_val_test(100, n_val, n_test)

        runner = ExperimentRunner(
            n_trials=self.config['n_trials'],
            hidden_sizes=[10],
            random_seed=self.config['random_seed']
        )

        print("\nRunning training size variation experiments...")
        return runner.run_training_size_experiment(train_sizes, X_val, y_val, X_test, y_test, data_gen)

    def _run_standard_experiment(self, data_gen):
        n_train = self.config['n_train']
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            data_gen.generate_train_val_test(n_train, self.config['n_val'], self.config['n_test'])

        runner = ExperimentRunner(
            n_trials=self.config['n_trials'],
            hidden_sizes=self.config['hidden_sizes'],
            random_seed=self.config['random_seed']
        )

        print("\nRunning standard experiments...")
        return runner.run_experiment(X_train, y_train, X_val, y_val, X_test, y_test)


class ResultsManager:
    def __init__(self, results, summary, output_dir):
        self.results = results
        self.summary = summary
        self.output_dir = output_dir

    def save_raw_results(self):
        clean_results = []
        for r in self.results:
            rc = r.copy()
            rc.pop('history', None)
            rc.pop('all_histories', None)
            clean_results.append(rc)

        path = os.path.join(self.output_dir, 'raw_results.json')
        with open(path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nRaw results saved to {path}")

    def save_summary(self):
        path = os.path.join(self.output_dir, 'summary.json')
        with open(path, 'w') as f:
            json.dump(self.summary, f, indent=2)
        print(f"Summary saved to {path}")

    def run_and_save_analysis(self):
        print("\nPerforming statistical analysis...")
        analyzer = StatisticalAnalyzer(alpha=0.05)
        comparisons = analyzer.compare_all_methods(self.results)
        stats_path = os.path.join(self.output_dir, 'statistical_tests.txt')
        analyzer.print_comparison_report(comparisons, stats_path)

    def create_and_save_visualitzations(self, mode):
        print("\nCreating visualizations...")
        viz = Visualizer()
        if mode == 'train_size':
            viz.plot_training_size_effect(self.results, os.path.join(self.output_dir, 'training_size_effect.png'))
        else:
            viz.plot_convergence_curves(self.results, os.path.join(self.output_dir, 'convergence_analysis.png'))
            viz.plot_performance_comparison(self.results, os.path.join(self.output_dir, 'performance_comparison.png'))
            viz.plot_time_comparison(self.summary, os.path.join(self.output_dir, 'time_comparison.png'))
            viz.plot_hidden_neurons_distribution(self.results, os.path.join(self.output_dir, 'hidden_distribution.png'))
            viz.create_summary_table(self.summary, os.path.join(self.output_dir, 'summary_table.txt'))

    def print_summary(self):
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        for method, stats in self.summary.items():
            print(f"\n{method}:")
            print(f"  Test MSE:  {stats['test_mse_mean']:.6f} ± {stats['test_mse_std']:.6f}")
            print(f"  Time:      {stats['time_mean']:.2f}s ± {stats['time_std']:.2f}s")
            print(f"  Hidden:    {int(stats['hidden_mode'])} neurons (most common)")
        
        print("\n" + "="*80)
        print(f"All results saved to '{self.output_dir}/' directory")
        print("="*80)

def get_experiment_config():
    print("="*80)
    print("EVOLUTIONARY ALGORITHMS FOR NEURAL NETWORK TRAINING")
    print("="*80)
    print("\nChoose experiment mode:")
    print("  1. Quick test (2 trials, ~2-3 minutes)")
    print("  2. Full experiment (30 trials, ~30-60 minutes)")
    print("  3. Training size variation (5 sizes, ~15-20 minutes)")
    print("  4. Custom configuration")
    
    while True:
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            print("\n→ Quick test mode selected")
            return {
                'mode': 'standard',
                'n_trials': 2,
                'n_features': 5,
                'noise_level': 0.1,
                'hardness': 0.2,
                'n_train': 500,
                'n_val': 100,
                'n_test': 200,
                'hidden_sizes': [5, 10, 25, 50],
                'random_seed': 42,
                'output_dir': 'results'
            }
        elif choice == "2":
            print("\n→ Full experiment mode selected")
            return {
                'mode': 'standard',
                'n_trials': 30,
                'n_features': 5,
                'noise_level': 0.1,
                'hardness': 1.0,
                'n_train': 300,
                'n_val': 60,
                'n_test': 1000,
                'hidden_sizes': [5, 10, 15, 20, 25, 50],
                'random_seed': 42,
                'output_dir': 'results_standard_diff'
            }
        elif choice == "3":
            print("\n→ Training size variation experiment")
            return {
                'mode': 'train_size',
                'n_trials': 10,
                'n_features': 5,
                'noise_level': 0.1,
                'hardness': 1.0,
                'train_sizes': [50, 100, 200, 400, 800],
                'n_val': 40,
                'n_test': 1000,
                'random_seed': 42,
                'output_dir': 'results'
            }
        elif choice == "4":
            print("\n→ Custom configuration")
            try:
                n_trials = int(input("  Number of trials (default 30): ") or "30")
                n_train = int(input("  Training samples (default 200): ") or "200")
                hardness = float(input("  Problem hardness (0.5=easy, 1.0=normal, 2.0=hard): ") or "1.0")
                hidden_input = input("  Hidden sizes (e.g., 5,10,15,20): ").strip()
                hidden_sizes = [int(x) for x in hidden_input.split(",")] if hidden_input else [5, 10, 15, 20]
                
                return {
                    'mode': 'standard',
                    'n_trials': n_trials,
                    'n_features': 5,
                    'noise_level': 0.1,
                    'hardness': hardness,
                    'n_train': n_train,
                    'n_val': 40,
                    'n_test': 1000,
                    'hidden_sizes': hidden_sizes,
                    'random_seed': 42,
                    'output_dir': 'results'
                }
            except (ValueError, KeyboardInterrupt):
                print("  Invalid input. Please try again.")
                continue
        else:
            print("  Invalid choice. Please enter 1, 2, 3, or 4.")