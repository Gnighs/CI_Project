import numpy as np
import os
import json

from src.utils import ExperimentManager, ResultsManager, get_experiment_config

from src.data_generator import SyntheticDataGenerator
from src.experiment_runner import ExperimentRunner
from src.visualization import Visualizer
from src.statistical_analysis import StatisticalAnalyzer


def main():
    config = get_experiment_config()

    experiments = ExperimentManager(config)
    results = experiments.run()
    
    runner = ExperimentRunner(
        n_trials=config['n_trials'],
        hidden_sizes=config.get('hidden_sizes', [10]),
        random_seed=config['random_seed']
    )
    output_dir = config['output_dir']

    resultsManager = ResultsManager(output_dir)

    resultsManager.save_raw_results(results)

    summary = runner.summarize_results(results)
    resultsManager.save_summary(summary)

    resultsManager.analysis(results, summary, config['mode'])
    
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