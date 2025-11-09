import numpy as np
import os
import json

from src.utils import ConfigManager, ExperimentManager

from src.data_generator import SyntheticDataGenerator
from src.experiment_runner import ExperimentRunner
from src.visualization import Visualizer
from src.statistical_analysis import StatisticalAnalyzer


def main():
    config = ConfigManager().get_experiment_config()

    experiments = ExperimentManager(config)
    results = experiments.run()
    
    output_dir = config['output_dir']

    runner = ExperimentRunner(
        n_trials=config['n_trials'],
        hidden_sizes=config.get('hidden_sizes', [10]),
        random_seed=config['random_seed']
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
    
    if config['mode'] == 'train_size':
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