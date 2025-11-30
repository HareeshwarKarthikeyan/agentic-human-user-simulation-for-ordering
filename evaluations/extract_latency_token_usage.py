"""
Script to extract latency and token usage data from conversation logs.
Generates CSV files with message_id, guest_message, total_tokens, and total_latency_seconds
for each guest message in each experiment.
"""

import csv
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


def parse_usage_column(usage_str: str) -> int:
    """
    Parse the usage column to extract total tokens.
    
    Args:
        usage_str: String representation of usage data
        
    Returns:
        Total tokens used, or 0 if parsing fails
    """
    if not usage_str or usage_str == '' or usage_str == 'nan':
        return 0
    
    try:
        # Try to parse as dictionary
        if isinstance(usage_str, str):
            # Clean up the string
            usage_str = usage_str.strip()
            
            # Handle JSON-like strings
            if usage_str.startswith('{'):
                usage_dict = json.loads(usage_str)
            else:
                # Try to evaluate as Python literal
                usage_dict = ast.literal_eval(usage_str)
        else:
            usage_dict = usage_str
        
        # Extract total tokens
        if isinstance(usage_dict, dict):
            return usage_dict.get('total_tokens', 0)
    except (json.JSONDecodeError, SyntaxError, ValueError) as e:
        print(f"Warning: Could not parse usage data: {usage_str[:100]}...")
        return 0
    
    return 0


def extract_latency_exp1(row: Dict[str, Any]) -> float:
    """
    Extract latency for Experiment 1 from latency_ms column.
    
    Args:
        row: CSV row as dictionary
        
    Returns:
        Latency in seconds
    """
    try:
        latency_ms = float(row.get('latency_ms', 0))
        # Convert -1.0 (non-guest messages) to 0
        if latency_ms < 0:
            return 0.0
        return latency_ms / 1000.0  # Convert ms to seconds
    except (ValueError, TypeError):
        return 0.0


def extract_latency_other_exps(tool_calls_str: str) -> float:
    """
    Extract latency for Experiments 2-5 from tool_calls column.
    
    Args:
        tool_calls_str: String representation of tool_calls data
        
    Returns:
        Latency in seconds
    """
    if not tool_calls_str or tool_calls_str == '' or tool_calls_str == 'nan':
        return 0.0
    
    try:
        # Parse the tool_calls string
        if isinstance(tool_calls_str, str):
            tool_calls_str = tool_calls_str.strip()
            
            # Try to find total_latency_sec in the string
            # Look for pattern: 'total_latency_sec': <number>
            pattern = r"'total_latency_sec':\s*([\d.]+)"
            match = re.search(pattern, tool_calls_str)
            if match:
                return float(match.group(1))
            
            # Alternative pattern with double quotes
            pattern = r'"total_latency_sec":\s*([\d.]+)'
            match = re.search(pattern, tool_calls_str)
            if match:
                return float(match.group(1))
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not extract latency from tool_calls: {str(e)}")
    
    return 0.0


def process_experiment(exp_num: int, input_dir: Path, output_dir: Path):
    """
    Process all conversation logs for a single experiment.
    
    Args:
        exp_num: Experiment number (1-5)
        input_dir: Directory containing CSV log files
        output_dir: Directory to save output CSV files
    """
    # Create output directory if it doesn't exist
    exp_output_dir = output_dir / f"exp{exp_num}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files in the logs directory
    log_files = list(input_dir.glob("*.csv"))
    
    print(f"\nProcessing Experiment {exp_num}: Found {len(log_files)} log files")
    
    for log_file in log_files:
        # Extract conversation identifier from filename
        filename_parts = log_file.stem.replace('chat_with_guest_simulation_', '')
        if exp_num == 1:
            filename_parts = filename_parts.replace('llm_messages_log_', '')
        else:
            filename_parts = filename_parts.replace('agent_messages_log_', '')
        
        output_filename = f"{filename_parts}_latency_tokens.csv"
        output_path = exp_output_dir / output_filename
        
        # Process the log file
        guest_messages = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Check if this is a guest message
                    role = row.get('role', '')
                    name = row.get('name', '')
                    
                    is_guest = (role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] or
                               (isinstance(name, str) and name.startswith('guest')))
                    
                    if is_guest:
                        # Extract data
                        message_id = row.get('message_id', '')
                        content = row.get('content', '')
                        
                        # Extract tokens
                        usage_str = row.get('usage', '')
                        total_tokens = parse_usage_column(usage_str)
                        
                        # Extract latency based on experiment
                        if exp_num == 1:
                            latency_seconds = extract_latency_exp1(row)
                        else:
                            tool_calls_str = row.get('tool_calls', '')
                            latency_seconds = extract_latency_other_exps(tool_calls_str)
                        
                        guest_messages.append({
                            'message_id': message_id,
                            'guest_message': content,
                            'total_tokens': total_tokens,
                            'total_latency_seconds': latency_seconds
                        })
            
            # Write output CSV
            if guest_messages:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['message_id', 'guest_message', 'total_tokens', 'total_latency_seconds']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(guest_messages)
                
                print(f"  ✓ Processed {log_file.name}: {len(guest_messages)} guest messages")
            else:
                print(f"  ⚠ No guest messages found in {log_file.name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {log_file.name}: {str(e)}")


def main():
    """Main function to process all experiments."""
    # Define base paths
    base_dir = Path(__file__).parent.parent
    experiments_dir = base_dir / "experiments"
    output_dir = base_dir / "evaluations" / "results" / "latencies_and_token_usages"
    
    # Create main output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Extracting Latency and Token Usage Data")
    print("=" * 60)
    
    # Process each experiment
    for exp_num in range(1, 6):
        exp_dir = experiments_dir / f"exp{exp_num}" / "logs"
        
        if not exp_dir.exists():
            print(f"\n⚠ Warning: Experiment {exp_num} logs directory not found at {exp_dir}")
            continue
        
        process_experiment(exp_num, exp_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()