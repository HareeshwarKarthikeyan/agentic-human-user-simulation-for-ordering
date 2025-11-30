"""
Cost Analysis Module for Ablation Study
Analyzes token usage and latency costs across experiments.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class CostAnalyzer:
    """
    Analyzes computational costs (tokens and latency) across experiments.
    """
    
    # Pricing assumptions (example rates - adjust based on actual model)
    TOKEN_COST_PER_1K = 0.015  # $0.015 per 1000 tokens (example)
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize cost analyzer.
        
        Args:
            data_dir: Directory containing latency and token usage data
        """
        if data_dir is None:
            base_dir = Path(__file__).parent.parent
            data_dir = base_dir / "evaluations" / "results" / "latencies_and_token_usages"
        
        self.data_dir = data_dir
        self.experiment_data = {}
        
    def load_experiment_data(self, exp_num: int) -> Dict[str, List[Dict]]:
        """
        Load all conversation data for a specific experiment.
        
        Args:
            exp_num: Experiment number (1-5)
            
        Returns:
            Dictionary mapping conversation IDs to list of message data
        """
        exp_dir = self.data_dir / f"exp{exp_num}"
        
        if not exp_dir.exists():
            print(f"Warning: No data found for experiment {exp_num}")
            return {}
        
        conversations = {}
        
        for csv_file in exp_dir.glob("*_latency_tokens.csv"):
            # Extract conversation ID from filename
            conv_id = csv_file.stem.replace("_latency_tokens", "")
            
            messages = []
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        messages.append({
                            'message_id': row['message_id'],
                            'content': row['guest_message'],
                            'tokens': int(row['total_tokens']) if row['total_tokens'] else 0,
                            'latency': float(row['total_latency_seconds']) if row['total_latency_seconds'] else 0.0
                        })
                
                conversations[conv_id] = messages
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return conversations
    
    def calculate_conversation_costs(self, messages: List[Dict]) -> Dict[str, float]:
        """
        Calculate cost metrics for a single conversation.
        
        Args:
            messages: List of message data
            
        Returns:
            Dictionary of cost metrics
        """
        if not messages:
            return {
                'total_tokens': 0,
                'total_latency': 0.0,
                'num_messages': 0,
                'avg_tokens_per_message': 0,
                'avg_latency_per_message': 0.0,
                'token_cost': 0.0
            }
        
        # Filter out 'exit' messages for metrics calculation
        content_messages = [m for m in messages if m['content'].lower() != 'exit']
        
        total_tokens = sum(m['tokens'] for m in messages)
        total_latency = sum(m['latency'] for m in messages)
        num_messages = len(content_messages)
        
        return {
            'total_tokens': total_tokens,
            'total_latency': total_latency,
            'num_messages': num_messages,
            'avg_tokens_per_message': total_tokens / num_messages if num_messages > 0 else 0,
            'avg_latency_per_message': total_latency / num_messages if num_messages > 0 else 0,
            'token_cost': (total_tokens / 1000) * self.TOKEN_COST_PER_1K
        }
    
    def analyze_experiment(self, exp_num: int) -> Dict[str, any]:
        """
        Analyze all conversations in an experiment.
        
        Args:
            exp_num: Experiment number
            
        Returns:
            Aggregated statistics for the experiment
        """
        conversations = self.load_experiment_data(exp_num)
        
        if not conversations:
            return None
        
        all_metrics = []
        for conv_id, messages in conversations.items():
            metrics = self.calculate_conversation_costs(messages)
            metrics['conversation_id'] = conv_id
            all_metrics.append(metrics)
        
        # Calculate aggregate statistics
        total_tokens = [m['total_tokens'] for m in all_metrics]
        total_latencies = [m['total_latency'] for m in all_metrics]
        num_messages = [m['num_messages'] for m in all_metrics]
        token_costs = [m['token_cost'] for m in all_metrics]
        
        return {
            'experiment': exp_num,
            'num_conversations': len(conversations),
            'total_tokens': {
                'mean': np.mean(total_tokens),
                'std': np.std(total_tokens),
                'min': np.min(total_tokens),
                'max': np.max(total_tokens),
                'sum': np.sum(total_tokens)
            },
            'total_latency': {
                'mean': np.mean(total_latencies),
                'std': np.std(total_latencies),
                'min': np.min(total_latencies),
                'max': np.max(total_latencies),
                'sum': np.sum(total_latencies)
            },
            'num_messages': {
                'mean': np.mean(num_messages),
                'std': np.std(num_messages),
                'min': np.min(num_messages),
                'max': np.max(num_messages)
            },
            'token_cost': {
                'mean': np.mean(token_costs),
                'std': np.std(token_costs),
                'total': np.sum(token_costs)
            },
            'efficiency': {
                'tokens_per_message': np.sum(total_tokens) / np.sum(num_messages) if np.sum(num_messages) > 0 else 0,
                'latency_per_message': np.sum(total_latencies) / np.sum(num_messages) if np.sum(num_messages) > 0 else 0
            }
        }
    
    def compare_experiments(self) -> Dict:
        """
        Compare cost metrics across all experiments.
        
        Returns:
            Comparison results and analysis
        """
        results = {}
        
        # Analyze each experiment
        for exp_num in range(1, 6):
            exp_results = self.analyze_experiment(exp_num)
            if exp_results:
                results[f'exp{exp_num}'] = exp_results
        
        if not results:
            return None
        
        # Calculate relative costs (compared to baseline Exp1)
        if 'exp1' in results:
            baseline_tokens = results['exp1']['total_tokens']['mean']
            baseline_latency = results['exp1']['total_latency']['mean']
            baseline_cost = results['exp1']['token_cost']['mean']
            
            for exp_key in results:
                exp_data = results[exp_key]
                exp_data['relative_metrics'] = {
                    'token_increase': (exp_data['total_tokens']['mean'] - baseline_tokens) / baseline_tokens * 100,
                    'latency_increase': (exp_data['total_latency']['mean'] - baseline_latency) / baseline_latency * 100,
                    'cost_increase': (exp_data['token_cost']['mean'] - baseline_cost) / baseline_cost * 100
                }
        
        return results
    
    def get_cost_summary(self) -> str:
        """
        Generate just the summary table of computational costs.
        
        Returns:
            Summary table as string
        """
        results = self.compare_experiments()
        
        if not results:
            return "No data available for cost analysis."
        
        report = []
        report.append("\nCOST ANALYSIS SUMMARY")
        report.append("-" * 60)
        report.append(f"{'Experiment':<35} {'Avg Tokens':<12} {'Avg Latency(s)':<15}")
        report.append("-" * 60)
        
        exp_descriptions = {
            'exp1': "LLM Baseline",
            'exp2': "Guest Agent Basic",
            'exp3': "Guest + Order Tracking",
            'exp4': "Guest + Message Attrs",
            'exp5': "Full System"
        }
        
        for exp_key in sorted(results.keys()):
            exp_data = results[exp_key]
            exp_num = exp_data['experiment']
            desc = exp_descriptions.get(exp_key, exp_key)
            
            report.append(f"{f'Exp{exp_num}: {desc}':<35} "
                         f"{exp_data['total_tokens']['mean']:>11.1f} "
                         f"{exp_data['total_latency']['mean']:>14.2f}")
        
        report.append("-" * 60)
        return "\n".join(report)
    
    def generate_cost_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive cost analysis report.
        
        Args:
            output_file: Optional path to save the report
            
        Returns:
            Report as string
        """
        results = self.compare_experiments()
        
        if not results:
            return "No data available for cost analysis."
        
        report = []
        report.append("=" * 80)
        report.append("COST ANALYSIS REPORT - ABLATION STUDY")
        report.append("=" * 80)
        report.append("")
        
        # Experiment descriptions
        exp_descriptions = {
            'exp1': "LLM Baseline (no agents/tools)",
            'exp2': "Guest Agent with Basic Tools",
            'exp3': "Guest Agent + Order Tracking Agent",
            'exp4': "Guest Agent + Message Attributes Agent",
            'exp5': "Full System (all agents)"
        }
        
        # Summary table
        report.append("SUMMARY OF COMPUTATIONAL COSTS")
        report.append("-" * 80)
        report.append(f"{'Experiment':<40} {'Avg Tokens':<15} {'Avg Latency(s)':<15}")
        report.append("-" * 80)
        
        for exp_key in sorted(results.keys()):
            exp_data = results[exp_key]
            exp_num = exp_data['experiment']
            desc = exp_descriptions.get(exp_key, exp_key)
            
            report.append(f"{f'Exp{exp_num}: {desc}':<40} "
                         f"{exp_data['total_tokens']['mean']:>14.1f} "
                         f"{exp_data['total_latency']['mean']:>14.2f}")
        
        report.append("")
        
        # Detailed analysis per experiment
        report.append("DETAILED ANALYSIS BY EXPERIMENT")
        report.append("=" * 80)
        
        for exp_key in sorted(results.keys()):
            exp_data = results[exp_key]
            exp_num = exp_data['experiment']
            
            report.append("")
            report.append(f"Experiment {exp_num}: {exp_descriptions.get(exp_key, '')}")
            report.append("-" * 40)
            
            # Token usage
            report.append(f"Token Usage:")
            report.append(f"  Mean: {exp_data['total_tokens']['mean']:.1f} tokens per conversation")
            report.append(f"  Std:  {exp_data['total_tokens']['std']:.1f}")
            report.append(f"  Range: {exp_data['total_tokens']['min']:.0f} - {exp_data['total_tokens']['max']:.0f}")
            report.append(f"  Total: {exp_data['total_tokens']['sum']:.0f} tokens across all conversations")
            
            # Latency
            report.append(f"\nLatency:")
            report.append(f"  Mean: {exp_data['total_latency']['mean']:.2f} seconds per conversation")
            report.append(f"  Std:  {exp_data['total_latency']['std']:.2f}")
            report.append(f"  Range: {exp_data['total_latency']['min']:.2f} - {exp_data['total_latency']['max']:.2f}")
            
            # Messages
            report.append(f"\nMessages:")
            report.append(f"  Mean: {exp_data['num_messages']['mean']:.1f} guest messages per conversation")
            report.append(f"  Range: {exp_data['num_messages']['min']:.0f} - {exp_data['num_messages']['max']:.0f}")
            
            # Efficiency
            report.append(f"\nEfficiency:")
            report.append(f"  Tokens per message: {exp_data['efficiency']['tokens_per_message']:.1f}")
            report.append(f"  Latency per message: {exp_data['efficiency']['latency_per_message']:.2f} seconds")
            
            # Cost
            report.append(f"\nEstimated Cost:")
            report.append(f"  Mean cost per conversation: ${exp_data['token_cost']['mean']:.4f}")
            report.append(f"  Total cost for experiment: ${exp_data['token_cost']['total']:.2f}")
            
            # Relative metrics (if not baseline)
            if exp_key != 'exp1' and 'relative_metrics' in exp_data:
                report.append(f"\nRelative to Baseline (Exp1):")
                report.append(f"  Token increase: {exp_data['relative_metrics']['token_increase']:+.1f}%")
                report.append(f"  Latency increase: {exp_data['relative_metrics']['latency_increase']:+.1f}%")
                report.append(f"  Cost increase: {exp_data['relative_metrics']['cost_increase']:+.1f}%")
        
        report.append("")
        report.append("=" * 80)
        report.append("KEY INSIGHTS")
        report.append("=" * 80)
        
        # Calculate insights
        if 'exp1' in results and 'exp5' in results:
            exp1 = results['exp1']
            exp5 = results['exp5']
            
            report.append("")
            report.append("1. OVERALL COST-BENEFIT ANALYSIS:")
            report.append(f"   - Full system (Exp5) uses {exp5['relative_metrics']['token_increase']:+.1f}% more tokens than baseline")
            report.append(f"   - Full system has {exp5['relative_metrics']['latency_increase']:+.1f}% different latency")
            report.append(f"   - Estimated cost increase: {exp5['relative_metrics']['cost_increase']:+.1f}%")
            
            # Find most efficient configuration
            min_tokens = float('inf')
            most_efficient = None
            for exp_key, exp_data in results.items():
                if exp_data['efficiency']['tokens_per_message'] < min_tokens:
                    min_tokens = exp_data['efficiency']['tokens_per_message']
                    most_efficient = exp_key
            
            report.append("")
            report.append("2. EFFICIENCY ANALYSIS:")
            report.append(f"   - Most token-efficient: {most_efficient} ({min_tokens:.1f} tokens/message)")
            
            # Find fastest configuration
            min_latency = float('inf')
            fastest = None
            for exp_key, exp_data in results.items():
                if exp_data['total_latency']['mean'] < min_latency:
                    min_latency = exp_data['total_latency']['mean']
                    fastest = exp_key
            
            report.append(f"   - Fastest average response: {fastest} ({min_latency:.2f} seconds/conversation)")
        
        report.append("")
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        
        # Generate recommendations based on data
        if results:
            # Find best trade-off (example logic - adjust based on your priorities)
            report.append("Based on the cost-performance analysis:")
            report.append("")
            
            # Check if Exp3 or Exp4 provide good trade-offs
            if 'exp3' in results and 'exp4' in results:
                exp3_cost_increase = results['exp3'].get('relative_metrics', {}).get('cost_increase', 0)
                exp4_cost_increase = results['exp4'].get('relative_metrics', {}).get('cost_increase', 0)
                
                if abs(exp3_cost_increase) < 50 and abs(exp4_cost_increase) < 50:
                    report.append("- Experiments 3 and 4 offer moderate cost increases with targeted improvements")
                    report.append("- Consider these configurations for cost-conscious deployments")
            
            if 'exp5' in results:
                exp5_cost = results['exp5'].get('relative_metrics', {}).get('cost_increase', 0)
                if exp5_cost < 100:
                    report.append("- Full system (Exp5) provides complete functionality with acceptable cost overhead")
                else:
                    report.append("- Full system (Exp5) has significant cost overhead - use for high-value scenarios")
        
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated using token cost assumption: ${self.TOKEN_COST_PER_1K:.3f} per 1000 tokens")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Cost report saved to: {output_file}")
        
        return report_text
    
    def generate_cost_visualizations(self, output_dir: Optional[Path] = None):
        """
        Generate visualization plots for cost analysis.
        
        Args:
            output_dir: Directory to save plots
        """
        results = self.compare_experiments()
        
        if not results:
            print("No data available for visualization")
            return
        
        if output_dir is None:
            output_dir = self.data_dir.parent / "figures"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        experiments = sorted(results.keys())
        exp_nums = [results[exp]['experiment'] for exp in experiments]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Cost Analysis Across Experiments', fontsize=16)
        
        # 1. Token Usage Comparison
        ax = axes[0, 0]
        token_means = [results[exp]['total_tokens']['mean'] for exp in experiments]
        token_stds = [results[exp]['total_tokens']['std'] for exp in experiments]
        ax.bar(exp_nums, token_means, yerr=token_stds, capsize=5, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Average Tokens per Conversation')
        ax.set_title('Token Usage Comparison')
        ax.set_xticks(exp_nums)
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Latency Comparison
        ax = axes[0, 1]
        latency_means = [results[exp]['total_latency']['mean'] for exp in experiments]
        latency_stds = [results[exp]['total_latency']['std'] for exp in experiments]
        ax.bar(exp_nums, latency_means, yerr=latency_stds, capsize=5, color='lightcoral', edgecolor='darkred')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Average Latency (seconds)')
        ax.set_title('Latency Comparison')
        ax.set_xticks(exp_nums)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Efficiency Metrics
        ax = axes[1, 0]
        tokens_per_msg = [results[exp]['efficiency']['tokens_per_message'] for exp in experiments]
        latency_per_msg = [results[exp]['efficiency']['latency_per_message'] for exp in experiments]
        
        x = np.arange(len(exp_nums))
        width = 0.35
        
        ax.bar(x - width/2, tokens_per_msg, width, label='Tokens/Message', color='skyblue')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, latency_per_msg, width, label='Latency/Message (s)', color='lightcoral')
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Tokens per Message', color='skyblue')
        ax2.set_ylabel('Latency per Message (s)', color='lightcoral')
        ax.set_title('Efficiency Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_nums)
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='lightcoral')
        
        # 4. Cost Comparison
        ax = axes[1, 1]
        cost_means = [results[exp]['token_cost']['mean'] for exp in experiments]
        colors = ['gold' if i == 0 else 'lightgreen' if i == len(experiments)-1 else 'lightgray' 
                 for i in range(len(experiments))]
        bars = ax.bar(exp_nums, cost_means, color=colors, edgecolor='black')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Average Cost per Conversation ($)')
        ax.set_title('Estimated Cost Comparison')
        ax.set_xticks(exp_nums)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, cost in zip(bars, cost_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / 'cost_analysis_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cost analysis visualization saved to: {output_file}")
        
        plt.close()
        
        # Create a relative cost increase plot
        if 'exp1' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            relative_increases = []
            exp_labels = []
            
            for exp in experiments[1:]:  # Skip exp1 (baseline)
                if 'relative_metrics' in results[exp]:
                    relative_increases.append([
                        results[exp]['relative_metrics']['token_increase'],
                        results[exp]['relative_metrics']['latency_increase'],
                        results[exp]['relative_metrics']['cost_increase']
                    ])
                    exp_labels.append(f"Exp{results[exp]['experiment']}")
            
            if relative_increases:
                relative_increases = np.array(relative_increases)
                
                x = np.arange(len(exp_labels))
                width = 0.25
                
                ax.bar(x - width, relative_increases[:, 0], width, label='Token Increase', color='skyblue')
                ax.bar(x, relative_increases[:, 1], width, label='Latency Change', color='lightcoral')
                ax.bar(x + width, relative_increases[:, 2], width, label='Cost Increase', color='lightgreen')
                
                ax.set_xlabel('Experiment')
                ax.set_ylabel('Percentage Change from Baseline (%)')
                ax.set_title('Relative Cost Increases Compared to Baseline (Exp1)')
                ax.set_xticks(x)
                ax.set_xticklabels(exp_labels)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                plt.tight_layout()
                output_file = output_dir / 'relative_cost_increases.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Relative cost increase plot saved to: {output_file}")
                plt.close()


def main():
    """Main function to run cost analysis."""
    analyzer = CostAnalyzer()
    
    # Generate report
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "cost_analysis_report.txt"
    report = analyzer.generate_cost_report(report_file)
    print(report)
    
    # Generate visualizations
    analyzer.generate_cost_visualizations()
    
    # Also save results as JSON for further analysis
    results = analyzer.compare_experiments()
    if results:
        json_file = output_dir / "cost_analysis_data.json"
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            json.dump(convert_types(results), f, indent=2)
        print(f"Cost analysis data saved to: {json_file}")


if __name__ == "__main__":
    main()