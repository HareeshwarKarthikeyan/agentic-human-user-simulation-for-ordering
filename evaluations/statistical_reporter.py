"""
Statistical analysis and reporting module for ablation study results.
Generates statistical comparisons, significance tests, and visualizations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StatisticalReporter:
    """
    Generate statistical reports and comparisons for ablation experiments.
    """
    
    def __init__(self, results: Dict[int, List[Dict]]):
        """
        Initialize reporter with results.
        
        Args:
            results: Dictionary mapping experiment level to list of metric dicts
        """
        self.results = results
        self.metrics = ['PAS', 'BVS', 'TRA', 'DEI', 'CRRS']
    
    def generate_full_report(self) -> Dict:
        """
        Generate comprehensive statistical report.
        
        Returns:
            Dictionary containing all statistical analyses
        """
        report = {
            'summary_statistics': self._calculate_summary_statistics(),
            'pairwise_comparisons': self._calculate_pairwise_comparisons(),
            'effect_sizes': self._calculate_effect_sizes(),
            'improvement_analysis': self._analyze_improvements(),
            'metric_correlations': self._calculate_correlations()
        }
        
        return report
    
    def _calculate_summary_statistics(self) -> pd.DataFrame:
        """Calculate mean, std, and CI for each metric across experiments."""
        summary_data = []
        
        for exp_level in sorted(self.results.keys()):
            exp_results = self.results[exp_level]
            if not exp_results:
                continue
            
            df = pd.DataFrame(exp_results)
            
            for metric in self.metrics:
                if metric not in df.columns:
                    continue
                    
                values = df[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                sem = std_val / np.sqrt(len(values))
                
                # 95% confidence interval
                ci_lower = mean_val - 1.96 * sem
                ci_upper = mean_val + 1.96 * sem
                
                summary_data.append({
                    'Experiment': f'Config{exp_level}',
                    'Metric': metric,
                    'Mean': mean_val,
                    'Std': std_val,
                    'SEM': sem,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'N': len(values)
                })
        
        return pd.DataFrame(summary_data)
    
    def _calculate_pairwise_comparisons(self) -> Dict:
        """Perform pairwise statistical tests between experiments."""
        comparisons = {}
        
        for metric in self.metrics:
            metric_comparisons = {}
            
            # Key comparison: Config5 vs Config1
            if 5 in self.results and 1 in self.results:
                exp5_values = [r[metric] for r in self.results[5] if metric in r]
                exp1_values = [r[metric] for r in self.results[1] if metric in r]
                
                if exp5_values and exp1_values:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(exp5_values, exp1_values)
                    
                    # Mann-Whitney U test (non-parametric alternative)
                    u_stat, u_p_value = stats.mannwhitneyu(exp5_values, exp1_values)
                    
                    metric_comparisons['exp5_vs_exp1'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'u_statistic': u_stat,
                        'u_p_value': u_p_value,
                        'significant_at_05': p_value < 0.05,
                        'significant_at_01': p_value < 0.01
                    }
            
            # Progressive comparisons
            for i in range(1, 5):
                if i in self.results and (i+1) in self.results:
                    exp_i_values = [r[metric] for r in self.results[i] if metric in r]
                    exp_next_values = [r[metric] for r in self.results[i+1] if metric in r]
                    
                    if exp_i_values and exp_next_values:
                        t_stat, p_value = stats.ttest_ind(exp_next_values, exp_i_values)
                        
                        metric_comparisons[f'exp{i+1}_vs_exp{i}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_at_05': p_value < 0.05
                        }
            
            comparisons[metric] = metric_comparisons
        
        return comparisons
    
    def _calculate_effect_sizes(self) -> Dict:
        """Calculate Cohen's d effect sizes for key comparisons."""
        effect_sizes = {}
        
        for metric in self.metrics:
            metric_effects = {}
            
            # Config5 vs Config1 (main comparison)
            if 5 in self.results and 1 in self.results:
                exp5_values = [r[metric] for r in self.results[5] if metric in r]
                exp1_values = [r[metric] for r in self.results[1] if metric in r]
                
                if exp5_values and exp1_values:
                    # Cohen's d
                    mean_diff = np.mean(exp5_values) - np.mean(exp1_values)
                    pooled_std = np.sqrt((np.var(exp5_values) + np.var(exp1_values)) / 2)
                    
                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                    else:
                        cohens_d = 0
                    
                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        interpretation = "negligible"
                    elif abs(cohens_d) < 0.5:
                        interpretation = "small"
                    elif abs(cohens_d) < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                    
                    metric_effects['exp5_vs_exp1'] = {
                        'cohens_d': cohens_d,
                        'interpretation': interpretation,
                        'mean_difference': mean_diff
                    }
            
            effect_sizes[metric] = metric_effects
        
        return effect_sizes
    
    def _analyze_improvements(self) -> pd.DataFrame:
        """Analyze percentage improvements across experiments."""
        improvements = []
        
        for metric in self.metrics:
            # Calculate improvement from Config1 to each other configuration
            if 1 in self.results:
                exp1_mean = np.mean([r[metric] for r in self.results[1] if metric in r])
                
                for exp_level in [2, 3, 4, 5]:
                    if exp_level in self.results:
                        exp_mean = np.mean([r[metric] for r in self.results[exp_level] if metric in r])
                        
                        if exp1_mean > 0:
                            improvement_pct = ((exp_mean - exp1_mean) / exp1_mean) * 100
                        else:
                            improvement_pct = 0 if exp_mean == 0 else 100
                        
                        improvements.append({
                            'Metric': metric,
                            'Comparison': f'Config{exp_level} vs Config1',
                            'Config1_Mean': exp1_mean,
                            f'Config{exp_level}_Mean': exp_mean,
                            'Improvement_%': improvement_pct
                        })
        
        return pd.DataFrame(improvements)
    
    def _calculate_correlations(self) -> Dict:
        """Calculate correlations between metrics within each experiment."""
        correlations = {}
        
        for exp_level, exp_results in self.results.items():
            if not exp_results:
                continue
            
            df = pd.DataFrame(exp_results)
            
            # Select only metric columns that exist
            metric_cols = [m for m in self.metrics if m in df.columns]
            if len(metric_cols) < 2:
                continue
            
            # Calculate correlation matrix
            corr_matrix = df[metric_cols].corr()
            
            correlations[f'Config{exp_level}'] = corr_matrix.to_dict()
        
        return correlations
    
    def create_visualizations(self, output_dir: str = 'figures'):
        """Create and save visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Metric comparison across experiments
        self._plot_metric_comparison(output_path)
        
        # 2. Improvement heatmap
        self._plot_improvement_heatmap(output_path)
        
        # 3. Statistical significance plot
        self._plot_significance_matrix(output_path)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_metric_comparison(self, output_path: Path):
        """Create bar plot comparing metrics across experiments."""
        summary = self._calculate_summary_statistics()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics):
            metric_data = summary[summary['Metric'] == metric]
            
            ax = axes[i]
            experiments = metric_data['Experiment'].values
            means = metric_data['Mean'].values
            errors = metric_data['SEM'].values * 1.96  # 95% CI
            
            bars = ax.bar(experiments, means, yerr=errors, capsize=5)
            
            # Color code by experiment level
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'{metric} Score')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Metric Comparison Across Experiments', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'metric_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_heatmap(self, output_path: Path):
        """Create heatmap showing improvements from baseline."""
        improvements = self._analyze_improvements()
        
        # Pivot for heatmap
        pivot_data = improvements.pivot(index='Metric', columns='Comparison', values='Improvement_%')
        
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   annot_kws={'size': 24, 'weight': 'bold'},
                   cbar_kws={'label': 'Improvement %', 'shrink': 0.8})
        
        # Make axis labels bigger and bold - split x labels at 'vs'
        xlabels = [label.get_text().replace(' vs ', '\nvs\n') for label in ax.get_xticklabels()]
        ax.set_xticklabels(xlabels, fontsize=20, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight='bold')
        
        # Make axis labels bigger and bold
        ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight='bold')
        ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
        
        # Make colorbar label bigger
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Improvement %', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=18)
        
        plt.title('Percentage Improvement from Baseline (Config1)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'improvement_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_significance_matrix(self, output_path: Path):
        """Create matrix showing statistical significance of comparisons."""
        comparisons = self._calculate_pairwise_comparisons()
        
        # Create significance matrix for Config5 vs Config1
        sig_data = []
        for metric in self.metrics:
            if 'exp5_vs_exp1' in comparisons.get(metric, {}):
                comp = comparisons[metric]['exp5_vs_exp1']
                sig_data.append({
                    'Metric': metric,
                    'p_value': comp['p_value'],
                    'Significant': '***' if comp['p_value'] < 0.001 else 
                                 '**' if comp['p_value'] < 0.01 else
                                 '*' if comp['p_value'] < 0.05 else 'ns'
                })
        
        if sig_data:
            df_sig = pd.DataFrame(sig_data)
            
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
            
            # Create bar plot of p-values
            bars = ax.bar(df_sig['Metric'], -np.log10(df_sig['p_value']))
            
            # Color bars by significance
            for bar, sig in zip(bars, df_sig['Significant']):
                if sig == '***':
                    bar.set_color('#2ca02c')
                elif sig == '**':
                    bar.set_color('#1f77b4')
                elif sig == '*':
                    bar.set_color('#ff7f0e')
                else:
                    bar.set_color('#d62728')
            
            # Add significance threshold lines
            ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.5, label='p=0.05')
            ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.5, label='p=0.01')
            ax.axhline(y=-np.log10(0.001), color='g', linestyle='--', alpha=0.5, label='p=0.001')
            
            ax.set_ylabel('-log10(p-value)')
            ax.set_title('Statistical Significance: Config5 vs Config1')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'significance_plot.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_report(self, output_path: str = 'statistical_report.json'):
        """Save complete statistical report to file."""
        import json
        
        report = self.generate_full_report()
        
        # Convert to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        # Deep convert all nested structures
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Convert the entire report
        serializable_report = deep_convert(report)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"Statistical report saved to {output_path}")
        
        return report