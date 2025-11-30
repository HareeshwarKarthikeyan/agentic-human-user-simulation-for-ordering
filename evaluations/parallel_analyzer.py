"""
Parallel processing wrapper for ablation analysis to speed up LLM calls.
"""

import concurrent.futures
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

from .batched_state_extractor import BatchedStateExtractor
from .state_extractor import UnifiedStateExtractor
from .metrics_calculator import UnifiedMetricsCalculator


class ParallelAblationAnalyzer:
    """
    Processes multiple conversations in parallel to speed up analysis.
    """
    
    def __init__(self, llm_client, max_workers: int = 5):
        """
        Initialize parallel analyzer.
        
        Args:
            llm_client: LLM client for state extraction
            max_workers: Maximum number of parallel workers (default 5 to avoid rate limits)
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
    
    def analyze_conversations_parallel(self, 
                                      conversations: List[Dict],
                                      experiment_level: int,
                                      metric_logs_dir: str,
                                      extracted_states_dir: str) -> List[Dict]:
        """
        Analyze multiple conversations in parallel.
        
        Args:
            conversations: List of conversation data dicts
            experiment_level: Experiment level (1-5)
            metric_logs_dir: Directory for metric logs
            extracted_states_dir: Directory for extracted states
            
        Returns:
            List of metrics for each conversation
        """
        results = []
        
        # Determine if this experiment needs LLM
        needs_llm = experiment_level in [1, 2, 3, 4]
        
        if needs_llm:
            # Process in batches to avoid overwhelming the API
            batch_size = self.max_workers
            
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i+batch_size]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all conversations in batch
                    futures = []
                    for conv_data in batch:
                        future = executor.submit(
                            self._analyze_single_conversation,
                            conv_data,
                            experiment_level,
                            metric_logs_dir,
                            extracted_states_dir
                        )
                        futures.append((future, conv_data.get('id', 'unknown')))
                    
                    # Collect results as they complete
                    for future, conv_id in futures:
                        try:
                            metrics = future.result(timeout=60)  # 60 second timeout per conversation
                            results.append(metrics)
                            print(f".", end="", flush=True)  # Progress indicator
                        except concurrent.futures.TimeoutError:
                            print(f"\n   ⚠️  Timeout for {conv_id}")
                            results.append(self._empty_metrics(conv_id, experiment_level))
                        except Exception as e:
                            print(f"\n   ⚠️  Error in {conv_id}: {str(e)[:50]}")
                            results.append(self._empty_metrics(conv_id, experiment_level))
        else:
            # For non-LLM experiments (exp5), process sequentially as they're fast
            for conv_data in conversations:
                try:
                    metrics = self._analyze_single_conversation(
                        conv_data,
                        experiment_level,
                        metric_logs_dir,
                        extracted_states_dir
                    )
                    results.append(metrics)
                    print(f".", end="", flush=True)
                except Exception as e:
                    conv_id = conv_data.get('id', 'unknown')
                    print(f"\n   ⚠️  Error in {conv_id}: {str(e)[:50]}")
                    results.append(self._empty_metrics(conv_id, experiment_level))
        
        print()  # New line after progress dots
        return results
    
    def _analyze_single_conversation(self,
                                    conv_data: Dict,
                                    experiment_level: int,
                                    metric_logs_dir: str,
                                    extracted_states_dir: str) -> Dict:
        """
        Analyze a single conversation.
        """
        from .ablation_analyzer import AblationStudyAnalyzer
        
        # Create a local analyzer instance for thread safety
        analyzer = AblationStudyAnalyzer(
            self.llm_client,
            metric_logs_dir=metric_logs_dir,
            extracted_states_dir=extracted_states_dir
        )
        
        return analyzer.analyze_conversation(
            conv_data['log'],
            conv_data['target_order'],
            conv_data['persona'],
            experiment_level,
            conv_data.get('id', 'unknown')
        )
    
    def _empty_metrics(self, conv_id: str, experiment_level: int) -> Dict:
        """
        Return empty metrics for failed conversations.
        """
        return {
            'conversation_id': conv_id,
            'experiment_level': experiment_level,
            'PAS': 0.0,
            'BVS': 0.0,
            'ORA': 0.0,
            'STA': 0.0,
            'DEI': 0.0,
            'CRRS': 0.0,
            'error': True
        }