"""
Main analyzer module for ablation study comparison.
Coordinates state extraction, metrics calculation, and analysis across experiments.
"""

import json
import sys
import pandas as pd
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
from .state_extractor import UnifiedStateExtractor
from .batched_state_extractor import BatchedStateExtractor
from .final_state_extractor import FinalStateExtractor
from .metrics_calculator import UnifiedMetricsCalculator
from .cost_analyzer import CostAnalyzer


class AblationStudyAnalyzer:
    """
    Main analyzer for comparing ablation experiments.
    """
    
    def __init__(self, llm_client=None, metric_logs_dir: str = "evaluations/metric_logs", 
                 extracted_states_dir: str = "evaluations/extracted_states"):
        """
        Initialize the analyzer.
        
        Args:
            llm_client: LLM client for state inference (required for exp 1-2)
            metric_logs_dir: Directory to save individual metric CSV files
            extracted_states_dir: Directory to save extracted states
        """
        self.llm_client = llm_client
        self.results = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.metric_logs_dir = Path(metric_logs_dir)
        self.extracted_states_dir = Path(extracted_states_dir)
        # Track which aggregate files have been initialized for this run
        self._aggregate_files_initialized = {
            'metrics': set(),  # Track initialized metrics aggregate files
            'states': set()    # Track initialized states aggregate files
        }
    
    def analyze_conversation(self, 
                           conversation_log: Dict,
                           target_order: Dict,
                           persona: Dict,
                           experiment_level: int,
                           conversation_id: str = None,
                           progress_bar=None,
                           use_extracted_states: bool = False) -> Dict[str, float]:
        """
        Analyze a single conversation for all metrics.
        
        Args:
            conversation_log: Full conversation log
            target_order: Target order specification
            persona: Guest persona description
            experiment_level: 1-5 indicating ablation level
            progress_bar: Optional tqdm progress bar for updates
            use_extracted_states: If True, load states from files instead of extracting
            
        Returns:
            Dictionary of metric scores
        """
        if use_extracted_states:
            # Load states from file and calculate metrics directly
            return self._analyze_with_extracted_states(
                conversation_log, target_order, persona, 
                experiment_level, conversation_id
            )
        
        # Initialize extractors and calculators for this experiment level
        state_extractor = UnifiedStateExtractor(experiment_level, self.llm_client)
        calculator = UnifiedMetricsCalculator(experiment_level, state_extractor)
        
        # Calculate all metrics
        metrics = calculator.calculate_all_metrics(
            conversation_log,
            target_order,
            persona
        )
        
        # Add metadata
        metrics['experiment_level'] = experiment_level
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['conversation_id'] = conversation_id or 'unknown'
        
        # Save metrics to individual CSV file
        if conversation_id:
            self._save_metrics_to_csv(metrics, experiment_level, conversation_id)
            
            # Also save extracted states
            self._save_extracted_states(
                conversation_log, 
                target_order, 
                persona, 
                state_extractor, 
                experiment_level, 
                conversation_id,
                progress_bar
            )
        
        return metrics
    
    def analyze_from_csv(self, csv_path: str, experiment_level: int,
                        test_case_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Analyze a conversation from CSV log file.
        
        Args:
            csv_path: Path to CSV conversation log
            experiment_level: 1-5 indicating ablation level
            test_case_data: Optional test case with target_order and persona
            
        Returns:
            Dictionary of metric scores
        """
        # Load conversation from CSV
        conversation_log = self._load_conversation_from_csv(csv_path)
        
        # Extract test case info if not provided
        if not test_case_data:
            test_case_data = self._extract_test_case_from_log(conversation_log)
        
        return self.analyze_conversation(
            conversation_log,
            test_case_data.get('target_order', {}),
            test_case_data.get('persona', {}),
            experiment_level
        )
    
    def run_batch_analysis(self, 
                          conversations_by_experiment: Dict[int, List[Dict]],
                          use_extracted_states: bool = False) -> Dict:
        """
        Run analysis on all conversations across all experiments.
        
        Args:
            conversations_by_experiment: Dict mapping experiment level to list of
                                       conversation data dicts with keys:
                                       - 'log': conversation log
                                       - 'target_order': target order spec
                                       - 'persona': persona description
            use_extracted_states: If True, skip state extraction and load from existing files
                                       
        Returns:
            Results dictionary with metrics for all conversations
        """
        results = {}
        
        # Calculate total conversations for overall progress
        total_conversations = sum(len(convs) for convs in conversations_by_experiment.values())
        
        print(f"\n📊 Analyzing {total_conversations} conversations across {len(conversations_by_experiment)} experiments")
        print("=" * 60)
        
        for exp_level in [1, 2, 3, 4, 5]:
            if exp_level not in conversations_by_experiment:
                continue
            
            exp_config = {
                1: "LLM Baseline (using LLM for state extraction)",
                2: "Guest Agent (using LLM for state extraction)",
                3: "Guest + Order Tracking (mixed extraction)",
                4: "Guest + Message Attributes (mixed extraction)",
                5: "Full System (using tool-based extraction)"
            }
            
            conversations = conversations_by_experiment[exp_level]
            print(f"\n🔬 Experiment {exp_level}: {exp_config.get(exp_level, 'Unknown')}")
            print(f"   Processing {len(conversations)} conversations...")
            
            if use_extracted_states:
                print("   📂 Using pre-extracted states from files")
            else:
                # Check if this experiment needs LLM
                needs_llm_order = exp_level in [1, 2, 4]
                needs_llm_behavioral = exp_level in [1, 2, 3]
                
                if needs_llm_order or needs_llm_behavioral:
                    print(f"   ⚡ Using LLM for: ", end="")
                    parts = []
                    if needs_llm_order:
                        parts.append("order tracking")
                    if needs_llm_behavioral:
                        parts.append("behavioral state")
                    print(" & ".join(parts))
                else:
                    print("   ⚡ Using tool-based extraction (fast)")
            
            exp_results = []
            
            # Track LLM call statistics
            total_llm_calls = 0
            successful_convs = 0
            failed_convs = []
            
            # Use tqdm for progress bar
            with tqdm(conversations, desc=f"   Exp{exp_level}", leave=True, 
                     ncols=100, position=0, file=sys.stdout) as pbar:
                for i, conv_data in enumerate(pbar):
                    try:
                        conv_id = conv_data.get('id', f'conv_{i}')
                        
                        # Update progress bar description with current conversation (less frequent updates)
                        if i % 5 == 0:  # Update every 5 conversations
                            pbar.set_description(f"   Exp{exp_level} [{conv_id[:15]}...]")
                        
                        metrics = self.analyze_conversation(
                            conv_data['log'],
                            conv_data['target_order'],
                            conv_data['persona'],
                            exp_level,
                            conv_id,
                            pbar,
                            use_extracted_states
                        )
                        
                        exp_results.append(metrics)
                        successful_convs += 1
                        
                        # Count actual LLM calls (only if not using extracted states)
                        if not use_extracted_states:
                            if needs_llm_order:
                                total_llm_calls += 1  # Only 1 call for final order state
                            if needs_llm_behavioral:
                                messages = conv_data.get('log', {}).get('messages', [])
                                guest_msg_count = sum(1 for m in messages if m.get('role') in 
                                                    ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'])
                                total_llm_calls += guest_msg_count  # Still per-message for behavioral
                        
                    except Exception as e:
                        print(f"\n   ⚠️  Error in {conv_id}: {str(e)[:50]}...")
                        failed_convs.append(conv_id)
                        continue
            
            results[exp_level] = exp_results
            
            # Print summary
            print(f"   ✅ Completed: {successful_convs}/{len(conversations)} successful")
            if total_llm_calls > 0:
                print(f"   📊 LLM calls made: ~{total_llm_calls}")
            if failed_convs:
                print(f"   ❌ Failed: {', '.join(failed_convs[:3])}" + 
                      (f" and {len(failed_convs)-3} more" if len(failed_convs) > 3 else ""))
        
        self.results = results
        
        print("\n" + "=" * 60)
        print("✨ Analysis complete!")
        return results
    
    def _load_conversation_from_csv(self, csv_path: str) -> Dict:
        """Load and parse conversation from CSV file."""
        df = pd.read_csv(csv_path)
        
        messages = []
        for _, row in df.iterrows():
            message = {
                'message_id': row.get('message_id', ''),
                'role': row.get('role', ''),
                'content': row.get('content', ''),
                'name': row.get('name', ''),
                'timestamp': row.get('timestamp', '')
            }
            
            # Parse tool calls if present
            if 'tool_calls' in row and pd.notna(row['tool_calls']):
                try:
                    # Handle string representation of tool calls
                    tool_calls_str = row['tool_calls']
                    if isinstance(tool_calls_str, str) and tool_calls_str.strip():
                        # For experiment 3-5, tool_calls is a complex nested dict as string
                        # We just need to check if it exists, not parse the full structure
                        message['tool_calls'] = tool_calls_str if tool_calls_str else {}
                    else:
                        message['tool_calls'] = {}
                except:
                    message['tool_calls'] = {}
            
            messages.append(message)
        
        return {'messages': messages}
    
    def _extract_test_case_from_log(self, conversation_log: Dict) -> Dict:
        """
        Attempt to extract test case information from conversation content.
        This is a fallback when test case data isn't provided.
        """
        # Look for patterns in messages that indicate the target order
        messages = conversation_log.get('messages', [])
        
        target_items = []
        persona_hints = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            
            # Extract food items (simple pattern matching)
            if 'chicken sandwich' in content:
                target_items.append({'name': 'Grilled Chicken Sandwich'})
            if 'french fries' in content or 'fries' in content:
                # Add as modifier to last item if exists
                if target_items:
                    target_items[-1]['modifiers'] = [{'modifier_option_name': 'French Fries'}]
            
            # Extract persona hints
            if msg.get('role') in ['agent', 'guest']:
                if len(content.split()) < 20:  # Short messages suggest efficiency
                    persona_hints.append('efficient')
                if 'busy' in content:
                    persona_hints.append('busy')
        
        # Construct minimal test case
        return {
            'target_order': {'items': target_items},
            'persona': {'description': ' '.join(persona_hints) or 'standard customer'}
        }
    
    def save_results(self, output_path: str):
        """Save analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
    
    def _analyze_with_extracted_states(self, 
                                      conversation_log: Dict,
                                      target_order: Dict,
                                      persona: Dict,
                                      experiment_level: int,
                                      conversation_id: str) -> Dict[str, float]:
        """
        Analyze conversation using pre-extracted states loaded from files.
        
        Args:
            conversation_log: Full conversation log
            target_order: Target order specification
            persona: Guest persona description
            experiment_level: 1-5 indicating ablation level
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Dictionary of metric scores
        """
        # Load states from the extracted states file
        states_file = self.extracted_states_dir / f"exp{experiment_level}" / f"{conversation_id}_states.csv"
        
        if not states_file.exists():
            raise FileNotFoundError(f"Extracted states file not found: {states_file}")
        
        # Load states from CSV
        states_df = pd.read_csv(states_file)
        
        # Convert states DataFrame to dictionary format expected by metrics calculator
        states_by_index = {}
        for _, row in states_df.iterrows():
            msg_idx = int(row['message_index'])
            
            # Parse order state (handle both single and double quotes)
            try:
                current_items = row['current_items_in_order']
                if pd.notna(current_items) and current_items != '[]':
                    # Replace single quotes with double quotes for valid JSON
                    if isinstance(current_items, str):
                        current_items = current_items.replace("'", '"')
                    current_items = json.loads(current_items)
                else:
                    current_items = []
            except:
                current_items = []
            
            try:
                target_items = row['target_items_to_order']
                if pd.notna(target_items) and target_items != '[]':
                    # Replace single quotes with double quotes for valid JSON
                    if isinstance(target_items, str):
                        target_items = target_items.replace("'", '"')
                    target_items = json.loads(target_items)
                else:
                    target_items = []
            except:
                target_items = []
            
            order_state = {
                'current_items_in_order': current_items,
                'target_items_to_order': target_items
            }
            
            # Parse behavioral state if present (check any behavioral attribute, not just is_ordering_complete)
            behavioral_state = {}
            has_behavioral = False
            
            # Check if any behavioral attribute is present
            if ('is_ordering_complete' in row and pd.notna(row['is_ordering_complete']) and row['is_ordering_complete']) or \
               ('menu_exploration_style' in row and pd.notna(row['menu_exploration_style']) and row['menu_exploration_style']) or \
               ('mood_tone' in row and pd.notna(row['mood_tone']) and row['mood_tone']) or \
               ('ordering_style' in row and pd.notna(row['ordering_style']) and row['ordering_style']):
                has_behavioral = True
            
            if has_behavioral:
                behavioral_state = {
                    'is_ordering_complete': row.get('is_ordering_complete', 'no') if pd.notna(row.get('is_ordering_complete')) else 'no',
                    'next_message_menu_exploration_style': row.get('menu_exploration_style', 'does_not_explore_menu') if pd.notna(row.get('menu_exploration_style')) else 'does_not_explore_menu',
                    'next_message_mood_tone': row.get('mood_tone', 'casual') if pd.notna(row.get('mood_tone')) else 'casual',
                    'next_message_ordering_style': row.get('ordering_style', 'all_at_once') if pd.notna(row.get('ordering_style')) else 'all_at_once'
                }
            
            states_by_index[msg_idx] = {
                'order_state': order_state,
                'behavioral_state': behavioral_state
            }
        
        # Create a mock state extractor that returns pre-loaded states
        class PreloadedStateExtractor:
            def __init__(self, states):
                self.states = states
                self.experiment_level = experiment_level
            
            def extract_state(self, conversation_log, message_index, target_order, persona):
                return self.states.get(message_index, {
                    'order_state': {'current_items_in_order': [], 'target_items_to_order': []},
                    'behavioral_state': {
                        'is_ordering_complete': 'no',
                        'next_message_menu_exploration_style': 'does_not_explore_menu',
                        'next_message_mood_tone': 'casual',
                        'next_message_ordering_style': 'all_at_once'
                    }
                })
            
            def extract_order_tracking_state(self, conversation_log, message_index):
                state = self.states.get(message_index, {})
                return state.get('order_state', {
                    'current_items_in_order': [],
                    'target_items_to_order': []
                })
            
            def extract_behavioral_state(self, conversation_log, message_index, persona):
                state = self.states.get(message_index, {})
                return state.get('behavioral_state', {
                    'is_ordering_complete': 'no',
                    'next_message_menu_exploration_style': 'does_not_explore_menu',
                    'next_message_mood_tone': 'casual',
                    'next_message_ordering_style': 'all_at_once'
                })
        
        # Use the preloaded state extractor with metrics calculator
        mock_extractor = PreloadedStateExtractor(states_by_index)
        calculator = UnifiedMetricsCalculator(experiment_level, mock_extractor)
        
        # Calculate all metrics
        metrics = calculator.calculate_all_metrics(
            conversation_log,
            target_order,
            persona
        )
        
        # Add metadata
        metrics['experiment_level'] = experiment_level
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['conversation_id'] = conversation_id or 'unknown'
        
        # Save metrics to CSV (still needed for metrics)
        if conversation_id:
            self._save_metrics_to_csv(metrics, experiment_level, conversation_id)
        
        return metrics
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for all experiments.
        
        Returns:
            DataFrame with mean metrics for each experiment
        """
        summary_data = []
        
        for exp_level, exp_results in self.results.items():
            if not exp_results:
                continue
            
            # Calculate means for each metric
            metrics_df = pd.DataFrame(exp_results)
            mean_metrics = metrics_df[['PAS', 'BVS', 'ORA', 'DEI', 'CRRS']].mean()
            
            summary_data.append({
                'Experiment': f"Exp{exp_level}",
                'PAS': mean_metrics['PAS'],
                'BVS': mean_metrics['BVS'],
                'ORA': mean_metrics['ORA'],
                'DEI': mean_metrics['DEI'],
                'CRRS': mean_metrics['CRRS'],
                'N': len(exp_results)
            })
        
        return pd.DataFrame(summary_data)
    
    def _save_metrics_to_csv(self, metrics: Dict, experiment_level: int, conversation_id: str):
        """
        Save metrics for a single conversation to a CSV file.
        
        Args:
            metrics: Dictionary of calculated metrics
            experiment_level: Experiment level (1-5)
            conversation_id: Unique identifier for the conversation
        """
        # Create directory structure
        exp_dir = self.metric_logs_dir / f"exp{experiment_level}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        # Create two formats: vertical (metric per row) and horizontal (all metrics in one row)
        
        # Vertical format (easier to read)
        vertical_file = exp_dir / f"{conversation_id}_metrics_detailed.csv"
        with open(vertical_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Description'])
            
            # Metric descriptions
            descriptions = {
                'PAS': 'Persona Adherence Score - How well the guest maintains their persona',
                'BVS': 'Behavioral Variance Score - Realistic fluctuations in behavior',
                'ORA': 'Order Restriction Adherence - Sticking to target order items',
                'STA': 'State Tracking Accuracy - Consistency of order state tracking',
                'DEI': 'Decision Explainability Index - Traceability of decisions',
                'CRRS': 'Composite Realism & Reliability Score - Overall quality',
                'experiment_level': 'Ablation experiment level (1-5)',
                'conversation_id': 'Unique conversation identifier',
                'timestamp': 'Analysis timestamp'
            }
            
            # Write metrics
            for key, value in metrics.items():
                if key in ['PAS', 'BVS', 'ORA', 'STA', 'DEI', 'CRRS']:
                    # Format numeric metrics to 4 decimal places
                    writer.writerow([key, f"{value:.4f}", descriptions.get(key, '')])
                else:
                    writer.writerow([key, value, descriptions.get(key, '')])
        
        # Horizontal format (better for aggregation)
        horizontal_file = exp_dir / f"{conversation_id}_metrics.csv"
        with open(horizontal_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'conversation_id', 'experiment_level', 'timestamp',
                'PAS', 'BVS', 'ORA', 'STA', 'DEI', 'CRRS'
            ])
            writer.writeheader()
            
            # Format numeric values
            row_data = {
                'conversation_id': metrics.get('conversation_id', ''),
                'experiment_level': metrics.get('experiment_level', ''),
                'timestamp': metrics.get('timestamp', ''),
                'PAS': f"{metrics.get('PAS', 0):.4f}",
                'BVS': f"{metrics.get('BVS', 0):.4f}",
                'ORA': f"{metrics.get('ORA', 0):.4f}",
                'STA': f"{metrics.get('STA', 0):.4f}",
                'DEI': f"{metrics.get('DEI', 0):.4f}",
                'CRRS': f"{metrics.get('CRRS', 0):.4f}"
            }
            writer.writerow(row_data)
        
        # Also create/append to an aggregate file for the experiment
        aggregate_file = exp_dir / "all_conversations_metrics.csv"
        
        # Determine if we should write header (first time writing to this file in this run)
        first_write = experiment_level not in self._aggregate_files_initialized['metrics']
        
        # Use 'w' mode for first write, 'a' mode for subsequent writes
        mode = 'w' if first_write else 'a'
        
        with open(aggregate_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'conversation_id', 'timestamp',
                'PAS', 'BVS', 'ORA', 'STA', 'DEI', 'CRRS'
            ])
            
            # Write header only on first write
            if first_write:
                writer.writeheader()
                self._aggregate_files_initialized['metrics'].add(experiment_level)
            
            # Write the data row (without experiment_level as it's redundant in per-experiment file)
            aggregate_row = {k: v for k, v in row_data.items() if k != 'experiment_level'}
            writer.writerow(aggregate_row)
    
    def _save_extracted_states(self, conversation_log: Dict, target_order: Dict,
                               persona: Dict, state_extractor: UnifiedStateExtractor,
                               experiment_level: int, conversation_id: str,
                               progress_bar=None):
        """
        Save extracted states for a conversation to CSV file.
        Uses batched extraction for LLM-based experiments to minimize API calls.
        
        Args:
            conversation_log: Full conversation log
            target_order: Target order specification
            persona: Guest persona
            state_extractor: State extractor instance (used for tool-based extraction)
            experiment_level: Experiment level (1-5)
            conversation_id: Unique conversation identifier
            progress_bar: Optional tqdm progress bar for updates
        """
        # Create directory structure
        exp_dir = self.extracted_states_dir / f"exp{experiment_level}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract states for all guest messages
        messages = conversation_log.get('messages', [])
        states_data = []
        
        # Determine if LLM is needed for this experiment
        needs_llm_order = experiment_level in [1, 2, 4]
        needs_llm_behavioral = experiment_level in [1, 2, 3]
        
        # Extract final order state ONCE for the entire conversation (much faster!)
        if needs_llm_order:
            final_extractor = FinalStateExtractor(experiment_level, self.llm_client)
            # Pass both full conversation AND target order for better extraction
            final_order_state = final_extractor.extract_final_order_state(conversation_log, target_order)
        else:
            final_order_state = None
        
        # Find the last guest message index
        last_guest_idx = -1
        for idx, msg in enumerate(messages):
            role = msg.get('role', '')
            content = msg.get('content', '')
            is_guest = (
                role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] or
                msg.get('name', '') in ['guest_simulation_agent', 'guest_simulation_llm'] or
                (role == 'agent' and 'guest' in msg.get('name', '').lower())
            )
            if is_guest and content != 'exit':
                last_guest_idx = idx
        
        # Use message-by-message extraction for LLM-based experiments
        if needs_llm_order or needs_llm_behavioral:
            # Process each message individually
            for idx, msg in enumerate(messages):
                role = msg.get('role', '')
                name = msg.get('name', '')
                is_guest = (
                    role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] or
                    name in ['guest_simulation_agent', 'guest_simulation_llm'] or
                    (role == 'agent' and 'guest' in name.lower())
                )
                
                if not is_guest or msg.get('content', '') == 'exit':
                    continue
                
                # Only use final order state for the LAST guest message
                if needs_llm_order:
                    if idx == last_guest_idx and final_order_state:
                        # This is the last guest message - use the extracted final state
                        order_state = final_order_state
                    else:
                        # All other messages get empty current_items but still need target_items
                        order_state = {'current_items_in_order': [], 'target_items_to_order': []}
                        # We'll set target_items_to_order below from target_order
                else:
                    order_state = None
                    
                # Still extract behavioral state per message (if needed)
                if needs_llm_behavioral:
                    behavioral_state = state_extractor.extract_behavioral_state(conversation_log, idx, str(persona))
                else:
                    behavioral_state = None
                
                if not order_state:
                    order_state = {'current_items_in_order': [], 'target_items_to_order': []}
                
                # Override target items with actual test case data
                if target_order:
                    target_items = []
                    for item in target_order.get('order_items', []):
                        target_item = {
                            'name': item.get('name', ''),
                            'quantity': item.get('quantity', 1),
                            'modifiers': []
                        }
                        for modifier in item.get('modifiers', []):
                            target_item['modifiers'].append(modifier.get('modifier_option_name', ''))
                        target_items.append(target_item)
                    order_state['target_items_to_order'] = target_items
                
                # Check for tool calls
                tool_calls = msg.get('tool_calls', '')
                has_tools = False
                if tool_calls and isinstance(tool_calls, str):
                    has_tools = bool(tool_calls.strip() and tool_calls != '{}')
                
                # Create state entry
                state_entry = {
                    'message_index': idx,
                    'message_id': msg.get('message_id', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'content_preview': msg.get('content', '')[:100],
                    'role': msg.get('role', ''),
                    'current_items_in_order': str(order_state.get('current_items_in_order', [])),
                    'target_items_to_order': str(order_state.get('target_items_to_order', [])),
                    'is_ordering_complete': behavioral_state.get('is_ordering_complete', '') if behavioral_state else '',
                    'menu_exploration_style': behavioral_state.get('next_message_menu_exploration_style', '') if behavioral_state else '',
                    'mood_tone': behavioral_state.get('next_message_mood_tone', '') if behavioral_state else '',
                    'ordering_style': behavioral_state.get('next_message_ordering_style', '') if behavioral_state else '',
                    'has_tool_calls': has_tools
                }
                
                states_data.append(state_entry)
        else:
            # Use regular extraction for tool-based experiments (exp3, exp5)
            for idx, msg in enumerate(messages):
                role = msg.get('role', '')
                name = msg.get('name', '')
                is_guest = (
                    role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] or
                    name in ['guest_simulation_agent', 'guest_simulation_llm'] or
                    (role == 'agent' and 'guest' in name.lower())
                )
                
                if not is_guest or msg.get('content', '') == 'exit':
                    continue
                
                # Extract states using tools
                order_state = state_extractor.extract_order_tracking_state(conversation_log, idx)
                behavioral_state = state_extractor.extract_behavioral_state(conversation_log, idx, str(persona))
                
                if not order_state:
                    order_state = {'current_items_in_order': [], 'target_items_to_order': []}
                
                # Override target items with actual test case data
                if target_order:
                    target_items = []
                    for item in target_order.get('order_items', []):
                        target_item = {
                            'name': item.get('name', ''),
                            'quantity': item.get('quantity', 1),
                            'modifiers': []
                        }
                        for modifier in item.get('modifiers', []):
                            target_item['modifiers'].append(modifier.get('modifier_option_name', ''))
                        target_items.append(target_item)
                    order_state['target_items_to_order'] = target_items
                
                # Check for tool calls
                tool_calls = msg.get('tool_calls', '')
                has_tools = False
                if tool_calls and isinstance(tool_calls, str):
                    has_tools = bool(tool_calls.strip() and tool_calls != '{}')
                
                # Create state entry
                state_entry = {
                    'message_index': idx,
                    'message_id': msg.get('message_id', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'content_preview': msg.get('content', '')[:100],
                    'role': msg.get('role', ''),
                    'current_items_in_order': str(order_state.get('current_items_in_order', [])),
                    'target_items_to_order': str(order_state.get('target_items_to_order', [])),
                    'is_ordering_complete': behavioral_state.get('is_ordering_complete', '') if behavioral_state else '',
                    'menu_exploration_style': behavioral_state.get('next_message_menu_exploration_style', '') if behavioral_state else '',
                    'mood_tone': behavioral_state.get('next_message_mood_tone', '') if behavioral_state else '',
                    'ordering_style': behavioral_state.get('next_message_ordering_style', '') if behavioral_state else '',
                    'has_tool_calls': has_tools
                }
                
                states_data.append(state_entry)
        
        # Save to CSV
        output_file = exp_dir / f"{conversation_id}_states.csv"
        
        if states_data:
            with open(output_file, 'w', newline='') as f:
                fieldnames = [
                    'message_index', 'message_id', 'timestamp', 'role', 'content_preview',
                    'current_items_in_order', 'target_items_to_order',
                    'is_ordering_complete', 'menu_exploration_style', 'mood_tone', 'ordering_style',
                    'has_tool_calls'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for state in states_data:
                    writer.writerow(state)
        
        # Also create an aggregate file for all conversations in this experiment
        aggregate_file = exp_dir / "all_conversations_states.csv"
        
        # Determine if we should write header (first time writing to this file in this run)
        first_write = experiment_level not in self._aggregate_files_initialized['states']
        
        # Use 'w' mode for first write, 'a' mode for subsequent writes
        mode = 'w' if first_write else 'a'
        
        if states_data:
            with open(aggregate_file, mode, newline='') as f:
                fieldnames = [
                    'conversation_id', 'message_index', 'message_id', 'timestamp', 'role',
                    'current_items_in_order', 'target_items_to_order',
                    'is_ordering_complete', 'menu_exploration_style', 'mood_tone', 'ordering_style',
                    'has_tool_calls'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header only on first write
                if first_write:
                    writer.writeheader()
                    self._aggregate_files_initialized['states'].add(experiment_level)
                
                # Write data rows with conversation_id
                for state in states_data:
                    row = {'conversation_id': conversation_id}
                    row.update({k: v for k, v in state.items() if k != 'content_preview'})
                    writer.writerow(row)