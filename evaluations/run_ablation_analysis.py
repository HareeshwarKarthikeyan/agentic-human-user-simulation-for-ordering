"""
Main script to run ablation analysis on conversation logs.
Analyzes conversations from all 4 experiment levels and generates reports.
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluations.ablation_analyzer import AblationStudyAnalyzer
from evaluations.statistical_reporter import StatisticalReporter
from evaluations.cost_analyzer import CostAnalyzer


class LLMClient:
    """
    LLM client wrapper for state inference using OpenAI.
    """
    def __init__(self, api_key: str = None, use_fallback: bool = False, model: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_fallback = use_fallback
        # Use GPT-4o as requested
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o')
        if model is None:
            print(f"Using {self.model} for evaluation")
        self.client = None
        
        if not self.api_key:
            print("Warning: OpenAI API key not found. Using fallback responses for state inference.")
            print("For accurate results, please set OPENAI_API_KEY environment variable.")
            self.use_fallback = True
        else:
            try:
                # Use OpenAI directly for simpler integration
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                print("Please install openai: pip install openai")
                self.use_fallback = True
        
    def complete(self, prompt: str) -> str:
        """
        Call LLM for completion using OpenAI.
        """
        if not self.use_fallback and self.client:
            try:
                # Call OpenAI with structured JSON response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise JSON extractor. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM inference error: {e}")
                # Try to extract specific info from the error for debugging
                if "api_key" in str(e).lower():
                    print("API key issue detected. Please check your OPENAI_API_KEY.")
                elif "rate" in str(e).lower():
                    print("Rate limit reached. Consider adding delays or using a different API key.")
        
        # Return a more detailed fallback that varies slightly
        import random
        base_response = {
            "current_items_in_order": [],
            "target_items_to_order": [],
            "is_ordering_complete": "no",
            "next_message_menu_exploration_style": "does_not_explore_menu",
            "next_message_mood_tone": "casual",
            "next_message_ordering_style": "all_at_once"
        }
        
        # Add some variation to detect if LLM is actually being used
        if random.random() > 0.5:
            base_response["next_message_mood_tone"] = random.choice(["casual", "friendly", "frustrated"])
            base_response["next_message_ordering_style"] = random.choice(["all_at_once", "one_by_one"])
        
        return json.dumps(base_response)


def load_conversation_logs(log_dir: str, pattern: str) -> List[Dict]:
    """
    Load conversation logs from directory.
    
    Args:
        log_dir: Directory containing log files
        pattern: File pattern to match (e.g., "*agent*" or "*llm*")
    
    Returns:
        List of conversation data dicts
    """
    log_path = Path(log_dir)
    conversations = []
    
    # Check if log_dir exists
    if not log_path.exists():
        print(f"  Log directory does not exist: {log_dir}")
        return conversations
    
    for csv_file in log_path.glob(pattern):
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Extract test case info from last row if it exists
            test_case_info = None
            if len(df) > 0 and df.iloc[-1]['role'] == 'system':
                test_case_row = df.iloc[-1]
                if 'content' in test_case_row:
                    test_case_info = json.loads(test_case_row['content'])
                df = df[:-1]  # Remove test case row from messages
            
            # Convert to conversation format
            messages = []
            for _, row in df.iterrows():
                message = {
                    'message_id': row.get('message_id', ''),
                    'role': row.get('role', ''),
                    'content': row.get('content', ''),
                    'name': row.get('name', ''),
                    'timestamp': row.get('timestamp', ''),
                    'tool_calls': row.get('tool_calls', {})
                }
                messages.append(message)
            
            conversation_data = {
                'id': csv_file.stem,
                'log': {'messages': messages},
                'target_order': test_case_info.get('target_order', {}) if test_case_info else {},
                'persona': test_case_info.get('guest_persona', {}) if test_case_info else {}
            }
            
            conversations.append(conversation_data)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return conversations


def main(use_extracted_states: bool = True):
    """
    Main function to run ablation analysis.
    
    Args:
        use_extracted_states: If True, skip state extraction and use existing extracted states
    """
    # Configuration
    BASE_DIR = Path(__file__).parent.parent  # Get to project root
    OUTPUT_DIR = BASE_DIR / "evaluations" / "results"
    
    # Experiment configurations based on your actual setup
    # Map experiment levels to log file patterns and descriptions
    experiment_configs = {
        1: {
            'log_dir': BASE_DIR / 'experiments/exp1/logs',
            'pattern': 'chat_with_guest_simulation_llm_messages_log_*.csv',
            'description': 'Exp1: LLM Baseline (single LLM with all info, no agents/tools)'
        },
        2: {
            'log_dir': BASE_DIR / 'experiments/exp2/logs',
            'pattern': 'chat_with_guest_simulation_agent_messages_log_*.csv',
            'description': 'Exp2: Guest Agent Only (no sub-agents, basic tools for name/persona/target/history)'
        },
        3: {
            'log_dir': BASE_DIR / 'experiments/exp3/logs',
            'pattern': 'chat_with_guest_simulation_agent_messages_log_*.csv',
            'description': 'Exp3: Guest Agent w. Order Tracking Agent'
        },
        4: {
            'log_dir': BASE_DIR / 'experiments/exp4/logs', 
            'pattern': 'chat_with_guest_simulation_agent_messages_log_*.csv',
            'description': 'Exp4: Guest Agent + Message Attributes Generation Agent'
        },
        5: {
            'log_dir': BASE_DIR / 'experiments/exp5/logs',
            'pattern': 'chat_with_guest_simulation_agent_messages_log_*.csv',
            'description': 'Exp5: Full System (Guest Agent w. Order Tracking Agent + Message Attributes Generation Agent)'
        }
    }
    
    print("=" * 60)
    print("ABLATION STUDY ANALYSIS")
    if use_extracted_states:
        print("(Using existing extracted states)")
    print("=" * 60)
    
    # Initialize LLM client for state inference (only if not using extracted states)
    llm_client = LLMClient() if not use_extracted_states else None
    
    # Initialize analyzer with metric logs and extracted states directories
    metric_logs_dir = Path(OUTPUT_DIR) / "metric_logs"
    extracted_states_dir = Path(OUTPUT_DIR) / "extracted_states"
    analyzer = AblationStudyAnalyzer(
        llm_client, 
        metric_logs_dir=str(metric_logs_dir),
        extracted_states_dir=str(extracted_states_dir)
    )
    
    # Load conversations for each experiment
    conversations_by_experiment = {}
    
    for exp_level, config in experiment_configs.items():
        print(f"\nLoading Experiment {exp_level}: {config['description']}")
        
        conversations = load_conversation_logs(str(config['log_dir']), config['pattern'])
        
        if conversations:
            conversations_by_experiment[exp_level] = conversations
            print(f"  Loaded {len(conversations)} conversations")
        else:
            print(f"  WARNING: No conversations found in {config['log_dir']} with pattern {config['pattern']}")
    
    # Check if we have data for key comparisons
    if not conversations_by_experiment:
        print("\nERROR: No conversation data found in any experiment directories!")
        print("Please run the experiments first to generate conversation logs.")
        return
    
    # Ensure we have at least exp1 and exp5 for meaningful comparison
    if 1 not in conversations_by_experiment:
        print("\nWARNING: No data for Experiment 1 (baseline). This is needed for comparison.")
    if 5 not in conversations_by_experiment:
        print("\nWARNING: No data for Experiment 5 (full system). This is the main system to evaluate.")
    
    if len(conversations_by_experiment) == 0:
        return
    
    # Run analysis
    print("\n" + "=" * 60)
    print("RUNNING ANALYSIS")
    print("=" * 60)
    
    results = analyzer.run_batch_analysis(conversations_by_experiment, use_extracted_states=use_extracted_states)
    
    # Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(str(OUTPUT_DIR / "ablation_results.json"))
    
    # Generate summary statistics
    summary_df = analyzer.get_summary_statistics()
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string())
    
    # Save summary to CSV
    summary_df.to_csv(str(OUTPUT_DIR / "summary_statistics.csv"), index=False)
    
    # Generate BVS interpretation table
    print("\n" + "=" * 60)
    print("BEHAVIORAL VARIANCE (BVS) INTERPRETATION")
    print("=" * 60)
    
    bvs_data = []
    for _, row in summary_df.iterrows():
        exp = row['Experiment']
        bvs_score = row['BVS']
        
        # Calculate approximate transition rate and interpretation
        # BVS peaks at 1.0 when transition rate = 20%
        # For BVS < 1.0, could be either side of peak
        
        if bvs_score <= 0.3:
            # Very low BVS - definitely too static
            transition_rate = bvs_score * 0.2 * 100  # In increasing region
            interpretation = "Too static (robotic)"
        elif bvs_score <= 0.5:
            # Low-moderate BVS - likely too static
            transition_rate = bvs_score * 0.2 * 100
            interpretation = "Low variance (somewhat static)"
        elif bvs_score <= 0.7:
            # Moderate BVS
            transition_rate = bvs_score * 0.2 * 100
            interpretation = "Moderate variance"
        elif bvs_score <= 0.85:
            # Good BVS - approaching optimal
            transition_rate = bvs_score * 0.2 * 100
            interpretation = "Good variance (realistic)"
        elif bvs_score <= 0.95:
            # Excellent BVS - near optimal
            transition_rate = bvs_score * 0.2 * 100
            interpretation = "Near-optimal realism"
        else:
            # Very high BVS - at or past optimal
            transition_rate = 20.0  # At peak
            interpretation = "Optimal behavioral realism"
        
        bvs_data.append({
            'Experiment': exp,
            'BVS Score': f"{bvs_score:.3f}",
            'Est. Transition Rate': f"~{transition_rate:.1f}%",
            'Interpretation': interpretation
        })
    
    import pandas as pd
    bvs_df = pd.DataFrame(bvs_data)
    print(bvs_df.to_string(index=False))
    
    # Save BVS interpretation to CSV
    bvs_df.to_csv(str(OUTPUT_DIR / "bvs_interpretation.csv"), index=False)
    
    # Generate statistical report
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    reporter = StatisticalReporter(results)
    report = reporter.generate_full_report()
    
    # Show key comparisons
    if 'pairwise_comparisons' in report:
        print("\nKey Statistical Comparisons (Exp5 vs Exp1):")
        print("-" * 40)
        
        for metric in ['PAS', 'BVS', 'ORA', 'STA', 'DEI', 'CRRS']:
            if metric in report['pairwise_comparisons']:
                comp = report['pairwise_comparisons'][metric].get('exp5_vs_exp1', {})
                if comp:
                    p_val = comp.get('p_value', 1.0)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"{metric:6s}: p={p_val:.4f} {sig}")
    
    # Show improvements
    if 'improvement_analysis' in report:
        improvements_df = pd.DataFrame(report['improvement_analysis'])
        exp5_improvements = improvements_df[improvements_df['Comparison'] == 'Exp5 vs Exp1']
        
        if not exp5_improvements.empty:
            print("\n" + "=" * 60)
            print("IMPROVEMENTS (Exp5 vs Exp1)")
            print("=" * 60)
            for _, row in exp5_improvements.iterrows():
                print(f"{row['Metric']:6s}: {row['Improvement_%']:+.1f}%")
    
    # Add cost analysis summary
    print("\n" + "=" * 60)
    print("COST ANALYSIS")
    print("=" * 60)
    try:
        cost_analyzer = CostAnalyzer()
        cost_summary = cost_analyzer.get_cost_summary()
        print(cost_summary)
    except Exception as e:
        print(f"Warning: Could not generate cost analysis: {e}")
    
    # Save statistical report
    reporter.save_report(str(OUTPUT_DIR / "statistical_report.json"))
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        reporter.create_visualizations(str(OUTPUT_DIR / "figures"))
        print("Visualizations created successfully!")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("Files generated:")
    print("  - ablation_results.json (raw results)")
    print("  - summary_statistics.csv (summary table)")
    print("  - statistical_report.json (full statistical analysis)")
    print("  - figures/ (visualization plots)")
    print("  - metric_logs/ (individual conversation metrics)")
    print("    └── exp[1-5]/ (organized by experiment)")
    print("        ├── <conversation_id>_metrics.csv (horizontal format)")
    print("        ├── <conversation_id>_metrics_detailed.csv (vertical format)")
    print("        └── all_conversations_metrics.csv (aggregate for experiment)")
    print("  - extracted_states/ (state extractions per conversation)")
    print("    └── exp[1-5]/ (organized by experiment)")
    print("        ├── <conversation_id>_states.csv (states for each message)")
    print("        └── all_conversations_states.csv (aggregate for experiment)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation analysis on conversation logs')
    parser.add_argument('--use-extracted-states', action='store_true',
                        help='Skip state extraction and use existing extracted states')
    
    args = parser.parse_args()
    main(use_extracted_states=args.use_extracted_states)