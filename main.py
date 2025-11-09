from src.utils import ExperimentManager, ResultsManager, get_experiment_config
from src.experiment_runner import ExperimentRunner

if __name__ == "__main__":
    config = get_experiment_config()

    experiments = ExperimentManager(config)
    
    runner = ExperimentRunner(
        n_trials=config['n_trials'],
        hidden_sizes=config.get('hidden_sizes', [10]),
        random_seed=config['random_seed']
    )

    results = experiments.run()
    summary = runner.summarize_results(results)

    resultsManager = ResultsManager(results, summary, output_dir=config['output_dir'])
    resultsManager.save_raw_results()
    resultsManager.save_summary()
    resultsManager.run_and_save_analysis()    
    resultsManager.create_and_save_visualitzations(config['mode'])
    resultsManager.print_summary()