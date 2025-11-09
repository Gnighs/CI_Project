import numpy as np
import os
import json
from src.data_generator import SyntheticDataGenerator
from src.experiment_runner import ExperimentRunner
from src.visualization import Visualizer
from src.statistical_analysis import StatisticalAnalyzer


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
                'hardness': 1.0,
                'n_train': 100,
                'n_val': 20,
                'n_test': 200,
                'hidden_sizes': [5, 10],
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
                'n_train': 200,
                'n_val': 40,
                'n_test': 1000,
                'hidden_sizes': [5, 10, 15, 20],
                'random_seed': 42,
                'output_dir': 'results'
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


def main():
    config = get_experiment_config()
    
    # Extract common configuration
    mode = config.get('mode', 'standard')
    n_trials = config['n_trials']
    n_features = config['n_features']
    noise_level = config['noise_level']
    hardness = config.get('hardness', 1.0)
    n_val = config['n_val']
    n_test = config['n_test']
    random_seed = config['random_seed']
    output_dir = config['output_dir']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"  Mode: {mode}")
    print(f"  Trials: {n_trials}")
    if mode == 'train_size':
        print(f"  Training sizes: {config['train_sizes']}")
    else:
        print(f"  Training samples: {config['n_train']}")
        print(f"  Hidden layer sizes: {config['hidden_sizes']}")
    print(f"  Validation samples: {n_val}")
    print(f"  Test samples: {n_test}")
    print(f"  Noise level: {noise_level}")
    print(f"  Problem hardness: {hardness}")
    print("="*80)
    
    # Generate data
    print("\nGenerating synthetic data...")
    data_gen = SyntheticDataGenerator(
        n_features=n_features,
        noise_level=noise_level,
        hardness=hardness,
        random_seed=random_seed
    )
    
    if mode == 'train_size':
        # Training size variation experiment
        train_sizes = config['train_sizes']
        _, (X_val, y_val), (X_test, y_test) = \
            data_gen.generate_train_val_test(100, n_val, n_test)
        
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        runner = ExperimentRunner(
            n_trials=n_trials,
            hidden_sizes=[10],
            random_seed=random_seed
        )
        
        print("\nRunning training size variation experiments...")
        results = runner.run_training_size_experiment(
            train_sizes, X_val, y_val, X_test, y_test, data_gen
        )
    else:
        # Standard experiment
        n_train = config['n_train']
        hidden_sizes = config['hidden_sizes']
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            data_gen.generate_train_val_test(n_train, n_val, n_test)
        
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        runner = ExperimentRunner(
            n_trials=n_trials,
            hidden_sizes=hidden_sizes,
            random_seed=random_seed
        )
        
        print("\nRunning experiments...")
        results = runner.run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
    
    # Save raw results (remove history for JSON serialization)
    results_for_json = []
    for r in results:
        r_copy = r.copy()
        if 'history' in r_copy:
            del r_copy['history']
        if 'all_histories' in r_copy:
            del r_copy['all_histories']
        results_for_json.append(r_copy)
    
    raw_results_path = os.path.join(output_dir, 'raw_results.json')
    with open(raw_results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nRaw results saved to {raw_results_path}")
    
    # Summarize results
    print("\nSummarizing results...")
    summary = runner.summarize_results(results)
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    analyzer = StatisticalAnalyzer(alpha=0.05)
    comparisons = analyzer.compare_all_methods(results)
    stats_path = os.path.join(output_dir, 'statistical_tests.txt')
    analyzer.print_comparison_report(comparisons, stats_path)
    
    # Visualization
    print("\nCreating visualizations...")
    viz = Visualizer()
    
    if mode == 'train_size':
        viz.plot_training_size_effect(
            results,
            os.path.join(output_dir, 'training_size_effect.png')
        )
    else:
        viz.plot_convergence_curves(
            results,
            os.path.join(output_dir, 'convergence_analysis.png')
        )
        viz.plot_performance_comparison(
            results,
            os.path.join(output_dir, 'performance_comparison.png')
        )
        viz.plot_time_comparison(
            summary,
            os.path.join(output_dir, 'time_comparison.png')
        )
        viz.plot_hidden_neurons_distribution(
            results,
            os.path.join(output_dir, 'hidden_distribution.png')
        )
        viz.create_summary_table(
            summary,
            os.path.join(output_dir, 'summary_table.txt')
        )
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  Test MSE:  {stats['test_mse_mean']:.6f} ± {stats['test_mse_std']:.6f}")
        print(f"  Time:      {stats['time_mean']:.2f}s ± {stats['time_std']:.2f}s")
        print(f"  Hidden:    {int(stats['hidden_mode'])} neurons (most common)")
    
    print("\n" + "="*80)
    print(f"All results saved to '{output_dir}/' directory")
    print("="*80)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()