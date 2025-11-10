import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_convergence_curves(self, results, output_file='convergence.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = {'MyGA': '#1f77b4', 'CMAES': '#ff7f0e', 'LBFGSB': '#2ca02c'}

        for method in ["MyGA", "CMAES"]:
            method_results = [r for r in results if r['method'] == method]
            if not method_results:
                continue
            
            all_histories = [r['history'] for r in method_results if 'history' in r and r['history']]
            if not all_histories:
                continue
            
            x_key, y_key = 'generations', 'best_fitness'
            min_len = min(len(h[x_key]) for h in all_histories)
            avg_fitness = np.mean([h[y_key][:min_len] for h in all_histories], axis=0)
            std_fitness = np.std([h[y_key][:min_len] for h in all_histories], axis=0)
            x_vals = list(range(1, min_len + 1))
            
            axes[0, 0].plot(x_vals, avg_fitness, label=method, color=colors[method], linewidth=2)
            axes[0, 0].fill_between(x_vals, avg_fitness - std_fitness, avg_fitness + std_fitness,
                                    alpha=0.2, color=colors[method])
        
        axes[0, 0].set_xlabel('Generation', fontsize=11)
        axes[0, 0].set_ylabel('Best Fitness (-MSE)', fontsize=11)
        axes[0, 0].set_title('Convergence: MyGA & CMAES', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        method = "LBFGSB"
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            all_histories = [r['history'] for r in method_results if 'history' in r and r['history']]
            if all_histories:
                x_key, y_key = 'iterations', 'fitness'
                min_len = min(len(h[x_key]) for h in all_histories)
                avg_fitness = np.mean([h[y_key][:min_len] for h in all_histories], axis=0)
                std_fitness = np.std([h[y_key][:min_len] for h in all_histories], axis=0)
                x_vals = list(range(1, min_len + 1))
                
                axes[0, 1].plot(x_vals, avg_fitness, label=method, color=colors[method], linewidth=2)
                axes[0, 1].fill_between(x_vals, avg_fitness - std_fitness, avg_fitness + std_fitness,
                                        alpha=0.2, color=colors[method])
        
        axes[0, 1].set_xlabel('Iteration', fontsize=11)
        axes[0, 1].set_ylabel('Best Fitness (-MSE)', fontsize=11)
        axes[0, 1].set_title('Convergence: LBFGSB', fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        for method in ["MyGA", "CMAES"]:
            method_results = [r for r in results if r['method'] == method]
            all_histories = [r['history'] for r in method_results if 'history' in r and r['history'] and 'diversity' in r['history']]
            if not all_histories:
                continue
            
            min_len = min(len(h['diversity']) for h in all_histories)
            avg_diversity = np.mean([h['diversity'][:min_len] for h in all_histories], axis=0)
            x_vals = list(range(1, min_len + 1))
            
            axes[1, 0].plot(x_vals, avg_diversity, label=method, color=colors[method], linewidth=2)
        
        axes[1, 0].set_xlabel('Generation', fontsize=11)
        axes[1, 0].set_ylabel('Population Diversity', fontsize=11)
        axes[1, 0].set_title('Diversity Over Time', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        methods = ["MyGA", "CMAES", "LBFGSB"]
        for method in methods:
            method_results = [r for r in results if r['method'] == method]
            final_test_mses = [-r['test_mse'] for r in method_results if 'test_mse' in r]
            
            if final_test_mses:
                min_val = min(final_test_mses)
                max_val = max(final_test_mses)
                bin_start = np.floor(min_val * 10) / 10
                bin_end = np.ceil(max_val * 10) / 10
                bins = np.linspace(bin_start, bin_end, 15)

                axes[1, 1].hist(final_test_mses, bins=bins, alpha=0.6, label=method, color=colors[method])

        axes[1, 1].set_xlabel('Final Fitness (-Test MSE)', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Final Fitness Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        axes[1, 1].set_xticks(np.round(bins, 2))
        axes[1, 1].set_xticklabels([f"{b:.2f}" for b in bins], rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence analysis to {output_file}")

        
    def plot_convergence(self, convergence_data, output_file='convergence.png'):
        plt.figure(figsize=self.figsize)
        
        for method, data in convergence_data.items():
            generations = data['generations']
            fitness = data['fitness']
            plt.plot(generations, fitness, label=method, linewidth=2)
        
        plt.xlabel('Generation / Iteration', fontsize=12)
        plt.ylabel('Best Fitness (Negative MSE)', fontsize=12)
        plt.title('Convergence Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved convergence plot to {output_file}")
    
    def plot_performance_comparison(self, results, output_file='performance.png'):
        methods = sorted(set([r['method'] for r in results]))
        test_mses = {method: [r['test_mse'] for r in results if r['method'] == method] 
                     for method in methods}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Box plot
        data_to_plot = [test_mses[method] for method in methods]
        bp = ax1.boxplot(data_to_plot, labels=methods, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Test MSE', fontsize=12)
        ax1.set_title('Test Error Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Violin plot
        positions = range(1, len(methods) + 1)
        parts = ax2.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(methods)
        ax2.set_ylabel('Test MSE', fontsize=12)
        ax2.set_title('Test Error Distribution (Violin)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance comparison to {output_file}")
    
    def plot_time_comparison(self, summary, output_file='time_comparison.png'):
        methods = list(summary.keys())
        times = [summary[m]['time_mean'] for m in methods]
        stds = [summary[m]['time_std'] for m in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Average Execution Time Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time comparison to {output_file}")
    
    def plot_hidden_neurons_distribution(self, results, output_file='hidden_distribution.png'):
        methods = sorted(set([r['method'] for r in results]))
        
        fig, axes = plt.subplots(1, len(methods), figsize=(14, 4))
        if len(methods) == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            hidden_sizes = [r['n_hidden'] for r in results if r['method'] == method]
            unique, counts = np.unique(hidden_sizes, return_counts=True)
            
            axes[idx].bar(unique, counts, color=f'C{idx}', alpha=0.7)
            axes[idx].set_xlabel('Number of Hidden Neurons', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'{method}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_xticks(unique)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved hidden neurons distribution to {output_file}")
    
    def create_summary_table(self, summary, output_file='summary_table.txt'):
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENTAL RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for method, stats in summary.items():
                f.write(f"\n{method}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Test MSE:  {stats['test_mse_mean']:.6f} ± {stats['test_mse_std']:.6f}\n")
                f.write(f"  Val MSE:   {stats['val_mse_mean']:.6f} ± {stats['val_mse_std']:.6f}\n")
                f.write(f"  Time:      {stats['time_mean']:.2f}s ± {stats['time_std']:.2f}s\n")
                f.write(f"  Hidden:    {stats['hidden_mean']:.1f} (mode: {stats['hidden_mode']})\n")
        
        print(f"Saved summary table to {output_file}")
    
    def plot_training_size_effect(self, train_size_results,
                                  output_file='training_size_effect.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        methods = sorted(set([r['method'] for r in train_size_results]))
        colors = {'MyGA': '#1f77b4', 'CMAES': '#ff7f0e', 'LBFGSB': '#2ca02c'}
        
        # Group by method and train size
        data_by_method = {m: {} for m in methods}
        for r in train_size_results:
            method = r['method']
            size = r.get('train_size', r.get('n_train', 0))
            if size not in data_by_method[method]:
                data_by_method[method][size] = []
            data_by_method[method][size].append(r['test_mse'])
        
        # Plot test error vs training size
        for method in methods:
            sizes = sorted(data_by_method[method].keys())
            means = [np.mean(data_by_method[method][s]) for s in sizes]
            stds = [np.std(data_by_method[method][s]) for s in sizes]
            
            axes[0].plot(sizes, means, 'o-', label=method,
                        color=colors.get(method, 'gray'), linewidth=2,
                        markersize=8)
            axes[0].fill_between(sizes,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2, color=colors.get(method, 'gray'))
        
        axes[0].set_xlabel('Training Set Size', fontsize=12)
        axes[0].set_ylabel('Test MSE', fontsize=12)
        axes[0].set_title('Effect of Training Size on Performance',
                         fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Plot execution time vs training size
        time_by_method = {m: {} for m in methods}
        for r in train_size_results:
            method = r['method']
            size = r.get('train_size', r.get('n_train', 0))
            if 'time' in r:
                if size not in time_by_method[method]:
                    time_by_method[method][size] = []
                time_by_method[method][size].append(r['time'])
        
        for method in methods:
            if not time_by_method[method]:
                continue
            sizes = sorted(time_by_method[method].keys())
            mean_times = [np.mean(time_by_method[method][s]) for s in sizes]
            
            axes[1].plot(sizes, mean_times, 'o-', label=method,
                        color=colors.get(method, 'gray'), linewidth=2,
                        markersize=8)
        
        axes[1].set_xlabel('Training Set Size', fontsize=12)
        axes[1].set_ylabel('Execution Time (s)', fontsize=12)
        axes[1].set_title('Computational Cost vs Training Size',
                         fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training size analysis to {output_file}")

