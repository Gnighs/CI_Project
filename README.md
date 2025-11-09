# CI-MAI: Evolutionary Computation Practical Work

## 1. Project Overview

**Chosen Problem:** **Training Neural Networks with Evolutionary Algorithms**

This project explores the use of Evolutionary Algorithms (EAs) as an alternative to traditional derivative-based methods for training Artificial Neural Networks (ANNs). We compare three optimization approaches:

- **Genetic Algorithm (GA)**: Classical evolutionary approach with selection, crossover, and mutation
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy, a more sophisticated EA
- **L-BFGS-B**: Quasi-Newton derivative-based method (baseline)

## 2. Requirements

The project is implemented using Python 3. All necessary libraries can be installed using:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy: Numerical computations
- scipy: Scientific computing and optimization
- cma: CMA-ES implementation
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization

## 3. Execution

To run the experiments, execute:

```bash
python main.py
```

You'll be presented with a simple menu:

```
Choose experiment mode:
  1. Quick test (2 trials, ~2-3 minutes)
  2. Full experiment (30 trials, ~30-60 minutes)
  3. Training size variation (5 sizes, ~15-20 minutes)
  4. Custom configuration
```

- **Quick test** (Option 1): Verifies everything works with minimal time investment
- **Full experiment** (Option 2): Complete statistical analysis needed for your report, includes convergence curves, diversity plots, and fitness distributions
- **Training size variation** (Option 3): Tests how methods perform with different amounts of training data (50, 100, 200, 400, 800 samples)
- **Custom** (Option 4): Lets you specify the number of trials, problem hardness, and other parameters

The program will:
1. Generate synthetic regression data with controlled noise
2. Run multiple independent trials for each optimization method (GA, CMA-ES, L-BFGS-B)
3. Test multiple network architectures
4. Perform statistical analysis comparing all methods
5. Generate visualizations and save results to the `results/` directory

## 4. Output

All results are saved in the `results/` directory:

**Standard Experiments (Options 1, 2, 4):**
- `raw_results.json`: Complete experimental data
- `summary.json`: Aggregated statistics for each method
- `summary_table.txt`: Human-readable summary table
- `statistical_tests.txt`: Statistical comparison results
- `convergence_analysis.png`: **NEW!** Convergence curves, diversity over time, and fitness distributions
- `performance_comparison.png`: Box and violin plots of test errors
- `time_comparison.png`: Execution time comparison
- `hidden_distribution.png`: Frequency of selected architectures

**Training Size Variation (Option 3):**
- `training_size_effect.png`: Performance and computational cost vs training size

## 5. Project Architecture

```
CI_Project/
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── main.py                     # Main experiment runner
├── report.tex                  # LaTeX report (Introduction and Problem Setup)
│
├── src/                        # Source code directory
│   ├── simple_mlp.py           # Multi-Layer Perceptron implementation
│   ├── optimizers.py           # MyGA, CMA-ES, and L-BFGS-B implementations
│   ├── data_generator.py       # Synthetic data generation
│   ├── experiment_runner.py    # Experiment orchestration and timing
│   ├── visualization.py        # Plotting functions
│   └── statistical_analysis.py # Statistical tests and comparisons
│
└── results/                    # Output directory (created on first run)
    ├── raw_results.json
    ├── summary.json
    ├── summary_table.txt
    ├── statistical_tests.txt
    └── *.png (various plots)
```

## 6. Reproducibility

All experiments use fixed random seeds to ensure reproducibility. The base seed is set to 42, with appropriate offsets for different trials and data splits.