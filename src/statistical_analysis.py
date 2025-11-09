import numpy as np
from scipy import stats

class StatisticalAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def wilcoxon_test(self, data1, data2):
        statistic, p_value = stats.wilcoxon(data1, data2)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def t_test(self, data1, data2, paired=False):
        if paired:
            statistic, p_value = stats.ttest_rel(data1, data2)
        else:
            statistic, p_value = stats.ttest_ind(data1, data2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def mann_whitney_test(self, data1, data2):
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def compare_all_methods(self, results):
        methods = sorted(set([r['method'] for r in results]))
        test_mses = {method: np.array([r['test_mse'] for r in results if r['method'] == method]) 
                     for method in methods}
        
        comparisons = {}
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                key = f"{method1}_vs_{method2}"
                
                # Wilcoxon signed-rank test (paired)
                wilcoxon = self.wilcoxon_test(test_mses[method1], test_mses[method2])
                
                # Paired t-test
                ttest = self.t_test(test_mses[method1], test_mses[method2], paired=True)
                
                # Mann-Whitney U test (unpaired)
                mannwhitney = self.mann_whitney_test(test_mses[method1], test_mses[method2])
                
                comparisons[key] = {
                    'wilcoxon': wilcoxon,
                    'paired_ttest': ttest,
                    'mannwhitney': mannwhitney,
                    'mean_diff': np.mean(test_mses[method1]) - np.mean(test_mses[method2]),
                    'effect_size': self._cohen_d(test_mses[method1], test_mses[method2])
                }
        
        return comparisons
    
    def _cohen_d(self, data1, data2):
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(data1) - np.mean(data2)) / pooled_std
    
    def print_comparison_report(self, comparisons, output_file='statistical_tests.txt'):
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            for comparison_name, results in comparisons.items():
                f.write(f"\n{comparison_name}:\n")
                f.write("-" * 60 + "\n")
                
                f.write(f"  Mean Difference: {results['mean_diff']:.6f}\n")
                f.write(f"  Cohen's d (Effect Size): {results['effect_size']:.4f}\n\n")
                
                f.write("  Wilcoxon Signed-Rank Test (paired):\n")
                f.write(f"    p-value: {results['wilcoxon']['p_value']:.6f}\n")
                f.write(f"    Significant: {results['wilcoxon']['significant']}\n\n")
                
                f.write("  Paired t-test:\n")
                f.write(f"    p-value: {results['paired_ttest']['p_value']:.6f}\n")
                f.write(f"    Significant: {results['paired_ttest']['significant']}\n\n")
                
                f.write("  Mann-Whitney U Test (unpaired):\n")
                f.write(f"    p-value: {results['mannwhitney']['p_value']:.6f}\n")
                f.write(f"    Significant: {results['mannwhitney']['significant']}\n")
                f.write("\n")
        
        print(f"Saved statistical analysis to {output_file}")

