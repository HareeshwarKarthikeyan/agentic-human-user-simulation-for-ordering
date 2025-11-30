"""
Metrics calculation module for evaluating guest simulation quality.
Calculates PAS, BVS, ORA, DEI, and composite scores.

Experiment Structure:
- Exp1: LLM baseline (no agents/tools)
- Exp2: Guest Agent only (basic tools)
- Exp3: Guest Agent + Order Tracking Agent
- Exp4: Guest Agent + Message Attributes Agent
- Exp5: Full system (all agents)

Note: STA (State Tracking Accuracy) removed as it's only applicable to 
experiments with order tracking (Exp3 & Exp5).
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .state_extractor import UnifiedStateExtractor


class UnifiedMetricsCalculator:
    """
    Calculates evaluation metrics for guest simulation conversations.
    """
    
    def __init__(self, experiment_level: int, state_extractor: UnifiedStateExtractor):
        """
        Initialize metrics calculator.
        
        Args:
            experiment_level: 1-5 indicating the ablation level
            state_extractor: State extractor instance for getting states
        """
        self.experiment_level = experiment_level
        self.state_extractor = state_extractor
        self.persona_attributes = self._load_persona_attributes()
    
    def calculate_all_metrics(self, 
                             conversation_log: Dict,
                             target_order: Dict,
                             persona: Dict) -> Dict[str, float]:
        """
        Calculate all metrics for a conversation.
        
        Args:
            conversation_log: Full conversation log
            target_order: Target order specification
            persona: Guest persona description
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'PAS': self.calculate_persona_adherence(conversation_log, persona),
            'BVS': self.calculate_behavioral_variance(conversation_log),
            'ORA': self.calculate_order_restriction_adherence(conversation_log, target_order),
            # STA removed - only applicable to experiments with order tracking (3 & 5)
            'DEI': self.calculate_decision_explainability(conversation_log)
        }
        
        metrics['CRRS'] = self.calculate_composite_score(metrics)
        return metrics
    
    def _load_persona_attributes(self) -> Dict:
        """Load persona attributes from the JSON file."""
        try:
            base_path = Path(__file__).parent.parent
            attributes_file = base_path / "test_case_generation" / "data" / "message_generation_attributes_for_personas.json"
            
            if attributes_file.exists():
                with open(attributes_file, 'r') as f:
                    data = json.load(f)
                    # Create a mapping from full_name to attributes
                    persona_map = {}
                    for entry in data.get('persona_attributes', []):
                        name = entry.get('full_name', '')
                        attrs = entry.get('message_generation_attributes', {})
                        persona_map[name] = attrs
                    return persona_map
        except Exception as e:
            print(f"Warning: Could not load persona attributes: {e}")
        
        return {}
    
    def calculate_persona_adherence(self, conversation_log: Dict, persona: Dict) -> float:
        """
        Calculate Persona Adherence Score (PAS).
        Measures how well the guest maintains their assigned persona based on
        predefined message generation attributes.
        """
        guest_messages_idx = self._get_guest_message_indices(conversation_log)
        
        # Filter out exit messages
        messages = conversation_log.get('messages', [])
        guest_messages_idx = [idx for idx in guest_messages_idx 
                              if idx < len(messages) and 
                              messages[idx].get('content', '').lower() != 'exit']
        
        if not guest_messages_idx:
            return 0.0
        
        # Extract guest name from persona
        guest_name = None
        if isinstance(persona, dict):
            guest_name = persona.get('full_name', '')
            if not guest_name and 'guest_persona' in persona:
                guest_name = persona['guest_persona'].get('full_name', '')
        
        # Get expected attributes for this persona
        expected_attrs = self.persona_attributes.get(guest_name, {})
        
        if not expected_attrs:
            # Fallback to old logic if no attributes found
            return self._calculate_persona_adherence_fallback(conversation_log, persona, guest_messages_idx)
        
        adherence_scores = []
        
        for idx in guest_messages_idx:
            behavioral_state = self.state_extractor.extract_behavioral_state(
                conversation_log, idx, str(persona)
            )
            
            if behavioral_state:
                # Calculate component scores
                component_scores = []
                
                # 1. Menu exploration style (weight: 0.25)
                expected_exploration = expected_attrs.get('next_message_menu_exploration_style', '')
                actual_exploration = behavioral_state.get('next_message_menu_exploration_style', '')
                if expected_exploration and actual_exploration:
                    exploration_score = 1.0 if expected_exploration == actual_exploration else 0.0
                    component_scores.append(('exploration', exploration_score, 0.25))
                
                # 2. Mood tone (weight: 0.25)
                expected_moods = expected_attrs.get('next_message_mood_tones', [])
                actual_mood = behavioral_state.get('next_message_mood_tone', '')
                if expected_moods and actual_mood:
                    # Check if actual mood is in the list of acceptable moods
                    mood_score = 1.0 if actual_mood in expected_moods else 0.0
                    component_scores.append(('mood', mood_score, 0.25))
                
                # 3. Ordering style (weight: 0.25)
                expected_ordering = expected_attrs.get('next_message_ordering_style', '')
                actual_ordering = behavioral_state.get('next_message_ordering_style', '')
                if expected_ordering and actual_ordering:
                    ordering_score = 1.0 if expected_ordering == actual_ordering else 0.0
                    component_scores.append(('ordering', ordering_score, 0.25))
                
                # 4. Order completion status (weight: 0.25)
                expected_complete = expected_attrs.get('is_ordering_complete', 'no')
                actual_complete = behavioral_state.get('is_ordering_complete', 'no')
                if expected_complete and actual_complete:
                    complete_score = 1.0 if expected_complete == actual_complete else 0.0
                    component_scores.append(('complete', complete_score, 0.25))
                
                # Calculate weighted average
                if component_scores:
                    total_weight = sum(weight for _, _, weight in component_scores)
                    weighted_sum = sum(score * weight for _, score, weight in component_scores)
                    message_score = weighted_sum / total_weight if total_weight > 0 else 0.0
                else:
                    message_score = 0.5  # Neutral if no components to compare
                
                adherence_scores.append(message_score)
        
        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0
    
    def _calculate_persona_adherence_fallback(self, conversation_log: Dict, persona: Dict, guest_messages_idx: List[int]) -> float:
        """
        Fallback persona adherence calculation using the old heuristic method.
        Used when persona attributes are not found in the JSON file.
        """
        adherence_scores = []
        persona_str = str(persona).lower()
        
        for idx in guest_messages_idx:
            behavioral_state = self.state_extractor.extract_behavioral_state(
                conversation_log, idx, persona_str
            )
            
            if behavioral_state:
                score = 0.0
                
                # Evaluate based on persona traits
                if 'busy professional' in persona_str or 'efficient' in persona_str:
                    # Busy professional should be efficient and direct
                    if behavioral_state.get('next_message_ordering_style') == 'all_at_once':
                        score += 0.33
                    if behavioral_state.get('next_message_menu_exploration_style') == 'does_not_explore_menu':
                        score += 0.33
                    if behavioral_state.get('next_message_mood_tone') in ['casual', 'frustrated']:
                        score += 0.34
                
                elif 'curious' in persona_str or 'explorer' in persona_str or 'adventurous' in persona_str:
                    # Curious personas should explore
                    if behavioral_state.get('next_message_menu_exploration_style') == 'explores_menu':
                        score += 0.5
                    if behavioral_state.get('next_message_mood_tone') in ['enthusiastic', 'casual']:
                        score += 0.5
                
                elif 'indecisive' in persona_str or 'uncertain' in persona_str:
                    # Indecisive personas might order one by one
                    if behavioral_state.get('next_message_ordering_style') == 'one_by_one':
                        score += 0.5
                    if behavioral_state.get('next_message_mood_tone') in ['confused', 'casual']:
                        score += 0.5
                
                elif 'family' in persona_str or 'group' in persona_str:
                    # Family/group orders tend to be more complex
                    if behavioral_state.get('next_message_ordering_style') == 'one_by_one':
                        score += 0.5
                    if behavioral_state.get('next_message_menu_exploration_style') == 'explores_menu':
                        score += 0.5
                
                else:
                    # Default scoring for other personas
                    score = 0.5  # Neutral score
                
                adherence_scores.append(score)
        
        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0
    
    def calculate_behavioral_variance(self, conversation_log: Dict) -> float:
        """
        Calculate Behavioral Variance Score (BVS).
        Measures realistic fluctuations in behavior.
        """
        guest_messages_idx = self._get_guest_message_indices(conversation_log)
        
        if len(guest_messages_idx) < 2:
            return 0.0
        
        behavioral_states = []
        for idx in guest_messages_idx:
            state = self.state_extractor.extract_behavioral_state(
                conversation_log, idx, ""
            )
            if state:
                behavioral_states.append(state)
        
        if len(behavioral_states) < 2:
            return 0.0
        
        # Calculate variance in each dimension
        ordering_styles = [s.get('next_message_ordering_style', '') for s in behavioral_states]
        exploration_styles = [s.get('next_message_menu_exploration_style', '') for s in behavioral_states]
        mood_tones = [s.get('next_message_mood_tone', '') for s in behavioral_states]
        
        # Count transitions (changes between consecutive states)
        def count_transitions(sequence):
            if len(sequence) < 2:
                return 0
            return sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        
        ordering_transitions = count_transitions(ordering_styles) / max(len(ordering_styles) - 1, 1)
        exploration_transitions = count_transitions(exploration_styles) / max(len(exploration_styles) - 1, 1)
        mood_transitions = count_transitions(mood_tones) / max(len(mood_tones) - 1, 1)
        
        # Realistic variance should be moderate (0.1-0.3 range is ideal)
        avg_transitions = (ordering_transitions + exploration_transitions + mood_transitions) / 3
        
        # Score peaks at 0.2 transitions (20% change rate)
        if avg_transitions <= 0.2:
            BVS = avg_transitions / 0.2
        else:
            BVS = max(0, 1 - (avg_transitions - 0.2) / 0.8)
        
        return max(0, min(1, BVS))
    
    def _normalize_item_name(self, name: str) -> str:
        """
        Normalize item names for consistent comparison.
        - Convert to lowercase
        - Remove special characters (keep only alphanumeric and spaces)
        - Remove extra whitespace
        - Remove common filler words
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove special characters but keep spaces
        normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common filler words that might appear inconsistently
        filler_words = ['the', 'a', 'an', 'with', 'and', 'or', 'of', 'in', 'on', 'at']
        words = normalized.split()
        words = [w for w in words if w not in filler_words]
        normalized = ' '.join(words)
        
        return normalized.strip()
    
    def calculate_order_restriction_adherence(self, 
                                             conversation_log: Dict, 
                                             target_order: Dict) -> float:
        """
        Calculate Order Restriction Adherence (ORA).
        Measures how well the guest sticks to their target order.
        Uses the LAST non-empty current_items_in_order in the conversation.
        Applies robust normalization to handle fuzzy matching from LLM judges.
        """
        guest_messages_idx = self._get_guest_message_indices(conversation_log)
        
        if not guest_messages_idx:
            return 0.0
        
        # Find the LAST non-empty current_items_in_order by checking all guest messages
        final_state = None
        for idx in reversed(guest_messages_idx):
            state = self.state_extractor.extract_order_tracking_state(conversation_log, idx)
            if state and state.get('current_items_in_order'):
                final_state = state
                break
        
        if not final_state or not final_state.get('current_items_in_order'):
            # Fallback to text extraction
            return self._calculate_ora_from_text(conversation_log, target_order)
        
        # Extract target items - handle both 'items' and 'target_order' keys
        # Also check for 'order_items' key
        target_items = set()
        target_data = target_order.get('items', target_order.get('target_order', target_order.get('order_items', [])))
        for item in target_data:
            if isinstance(item, dict):
                # Normalize the item name
                item_name = self._normalize_item_name(item.get('name', ''))
                modifiers = item.get('modifiers', [])
                
                # Handle modifiers - they could be strings or dicts
                if modifiers:
                    for modifier in modifiers:
                        if isinstance(modifier, dict):
                            modifier_name = self._normalize_item_name(
                                modifier.get('modifier_option_name', '') or 
                                modifier.get('name', '') or
                                modifier.get('modifier', '')
                            )
                        else:
                            modifier_name = self._normalize_item_name(str(modifier))
                        if modifier_name:
                            target_items.add(modifier_name)
                
                # Add the main item
                if item_name:
                    target_items.add(item_name)
        
        # Extract current items from state - handle both old and new field names
        current_items = set()
        for item in final_state.get('current_items_in_order', []):
            # Try both 'name' and 'item_name' fields, normalize them
            item_name = self._normalize_item_name(
                item.get('name', '') or 
                item.get('item_name', '') or
                item.get('item', '')
            )
            
            # Add the main item
            if item_name:
                current_items.add(item_name)
            
            # Handle modifiers - could be in 'modifiers' list or 'modifier' field
            modifiers = item.get('modifiers', [])
            if modifiers:
                for modifier in modifiers:
                    if isinstance(modifier, dict):
                        modifier_name = self._normalize_item_name(
                            modifier.get('name', '') or 
                            modifier.get('modifier_option_name', '') or
                            modifier.get('modifier', '')
                        )
                    else:
                        modifier_name = self._normalize_item_name(str(modifier))
                    if modifier_name:
                        current_items.add(modifier_name)
            
            # Also check single 'modifier' field for backward compatibility
            modifier = self._normalize_item_name(item.get('modifier', ''))
            if modifier:
                current_items.add(modifier)
        
        if not current_items and not target_items:
            return 0.0
        
        # Calculate precision and recall
        if len(current_items) == 0:
            precision = 0.0
        else:
            precision = len(current_items & target_items) / len(current_items)
        
        if len(target_items) == 0:
            recall = 0.0
        else:
            recall = len(current_items & target_items) / len(target_items)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        ORA = 2 * (precision * recall) / (precision + recall)
        return ORA
    
    def _extract_normalized_items_from_state(self, items_list: list) -> set:
        """
        Extract and normalize items from a state's current_items_in_order.
        Returns a set of normalized item names and modifiers.
        """
        normalized_items = set()
        
        for item in items_list:
            if isinstance(item, dict):
                # Main item name
                item_name = self._normalize_item_name(
                    item.get('name', '') or 
                    item.get('item_name', '') or
                    item.get('item', '')
                )
                if item_name:
                    normalized_items.add(item_name)
                
                # Handle modifiers
                modifiers = item.get('modifiers', [])
                if modifiers:
                    for modifier in modifiers:
                        if isinstance(modifier, dict):
                            modifier_name = self._normalize_item_name(
                                modifier.get('name', '') or 
                                modifier.get('modifier_option_name', '') or
                                modifier.get('modifier', '')
                            )
                        else:
                            modifier_name = self._normalize_item_name(str(modifier))
                        if modifier_name:
                            normalized_items.add(modifier_name)
                
                # Single modifier field
                modifier = self._normalize_item_name(item.get('modifier', ''))
                if modifier:
                    normalized_items.add(modifier)
        
        return normalized_items
    
    def calculate_state_tracking_accuracy(self, 
                                         conversation_log: Dict, 
                                         target_order: Dict) -> float:
        """
        Calculate State Tracking Accuracy (STA).
        Measures consistency and accuracy of order state tracking.
        Uses normalized item names for robust comparison.
        """
        guest_messages_idx = self._get_guest_message_indices(conversation_log)
        
        if not guest_messages_idx:
            return 0.0
        
        order_states = []
        for idx in guest_messages_idx:
            state = self.state_extractor.extract_order_tracking_state(
                conversation_log, idx
            )
            if state:
                order_states.append(state)
        
        if not order_states:
            return 0.0
        
        # Check monotonic growth (items shouldn't disappear) using normalized sets
        consistency_scores = []
        for i in range(1, len(order_states)):
            prev_items = self._extract_normalized_items_from_state(
                order_states[i-1].get('current_items_in_order', [])
            )
            curr_items = self._extract_normalized_items_from_state(
                order_states[i].get('current_items_in_order', [])
            )
            
            # Current should contain all previous items (or be the same)
            if len(prev_items) == 0 or prev_items.issubset(curr_items) or len(prev_items) == len(curr_items):
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.0)
        
        # Check final accuracy using normalized sets
        final_state = order_states[-1]
        
        # Extract and normalize target items
        target_items_normalized = set()
        target_data = target_order.get('items', target_order.get('target_order', target_order.get('order_items', [])))
        for item in target_data:
            if isinstance(item, dict):
                item_name = self._normalize_item_name(item.get('name', ''))
                if item_name:
                    target_items_normalized.add(item_name)
                
                # Handle modifiers
                modifiers = item.get('modifiers', [])
                if modifiers:
                    for modifier in modifiers:
                        if isinstance(modifier, dict):
                            modifier_name = self._normalize_item_name(
                                modifier.get('modifier_option_name', '') or 
                                modifier.get('name', '') or
                                modifier.get('modifier', '')
                            )
                        else:
                            modifier_name = self._normalize_item_name(str(modifier))
                        if modifier_name:
                            target_items_normalized.add(modifier_name)
        
        # Extract and normalize final items
        final_items_normalized = self._extract_normalized_items_from_state(
            final_state.get('current_items_in_order', [])
        )
        
        # Calculate accuracy based on normalized sets overlap
        if len(target_items_normalized) == 0:
            final_accuracy = 0.0
        else:
            # Accuracy is the proportion of target items that were correctly tracked
            correct_items = len(final_items_normalized & target_items_normalized)
            final_accuracy = correct_items / len(target_items_normalized)
        
        if consistency_scores:
            STA = (sum(consistency_scores) / len(consistency_scores) + final_accuracy) / 2
        else:
            STA = final_accuracy
        
        return STA
    
    def calculate_decision_explainability(self, conversation_log: Dict) -> float:
        """
        Calculate Decision Explainability Index (DEI).
        Measures how explainable/traceable decisions are.
        
        Exp1: No explainability (0)
        Exp2: Basic tools only (0.2)
        Exp3: Order tracking (0.5)
        Exp4: Message attributes (0.5)
        Exp5: Full explainability (1.0)
        """
        messages = conversation_log.get('messages', conversation_log)
        guest_messages = [msg for msg in messages if msg.get('role') in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm']]
        
        if not guest_messages:
            return 0.0
        
        if self.experiment_level == 5:
            # Full system: order tracking + behavioral
            explained_decisions = 0
            for msg in guest_messages:
                tool_calls_str = str(msg.get('tool_calls', {}))
                has_order = 'order_tracking' in tool_calls_str
                has_behavioral = 'message_generation' in tool_calls_str or 'message_attributes' in tool_calls_str
                explained_decisions += sum([has_order, has_behavioral])
            
            # Maximum 2 types of decisions per message
            DEI = min(1.0, explained_decisions / (len(guest_messages) * 2))
            
        elif self.experiment_level == 4:
            # Message attributes only
            explained_decisions = 0
            for msg in guest_messages:
                tool_calls_str = str(msg.get('tool_calls', {}))
                has_behavioral = 'message_generation' in tool_calls_str or 'message_attributes' in tool_calls_str
                explained_decisions += int(has_behavioral)
            
            DEI = min(0.5, explained_decisions / len(guest_messages) * 0.5)
            
        elif self.experiment_level == 3:
            # Order tracking only
            explained_decisions = 0
            for msg in guest_messages:
                tool_calls_str = str(msg.get('tool_calls', {}))
                has_order = 'order_tracking' in tool_calls_str
                explained_decisions += int(has_order)
            
            DEI = min(0.5, explained_decisions / len(guest_messages) * 0.5)
            
        elif self.experiment_level == 2:
            # Basic tools only (name, persona, target, history)
            explained_decisions = 0
            for msg in guest_messages:
                tool_calls_str = str(msg.get('tool_calls', {}))
                # Check for basic tools
                has_tools = any(tool in tool_calls_str for tool in ['get_name', 'get_persona', 'get_target', 'get_past_messages'])
                explained_decisions += int(has_tools)
            
            DEI = min(0.2, explained_decisions / len(guest_messages) * 0.2)
            
        else:
            # Experiment 1: No explainability
            DEI = 0.0
        
        return DEI
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted Composite Realism & Reliability Score (CRRS).
        """
        # Universal weights for all experiments
        weights = {
            'PAS': 0.25,  # Persona adherence - essential for realism
            'BVS': 0.20,  # Behavioral variance - adds naturalness
            'ORA': 0.35,  # Order accuracy - primary task (most critical)
            'DEI': 0.20   # Decision explainability - system validation
        }
        
        # Note: DEI will be 0 for experiments without explainability
        # This naturally adjusts the effective weights
        
        CRRS = sum(
            weights.get(metric, 0) * value 
            for metric, value in metrics.items() 
            if metric in weights
        )
        
        return CRRS
    
    def _get_guest_message_indices(self, conversation_log: Dict) -> List[int]:
        """Get indices of guest messages in the conversation."""
        messages = conversation_log.get('messages', conversation_log)
        return [i for i, msg in enumerate(messages) 
                if msg.get('role') in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] 
                or (isinstance(msg.get('name'), str) and msg.get('name').startswith('guest'))]
    
    def _calculate_ora_from_text(self, conversation_log: Dict, target_order: Dict) -> float:
        """Fallback ORA calculation using text extraction."""
        messages = conversation_log.get('messages', conversation_log)
        guest_messages = [msg for msg in messages if msg.get('role') in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm']]
        
        # Extract food items mentioned
        mentioned_items = set()
        for msg in guest_messages:
            content = msg.get('content', '').lower()
            # Simple extraction - could be enhanced with NLP
            if 'chicken' in content:
                mentioned_items.add('chicken')
            if 'sandwich' in content:
                mentioned_items.add('sandwich')
            if 'fries' in content or 'french fries' in content:
                mentioned_items.add('fries')
            if 'burger' in content:
                mentioned_items.add('burger')
            if 'pizza' in content:
                mentioned_items.add('pizza')
            if 'salad' in content:
                mentioned_items.add('salad')
            # Add more patterns as needed
        
        # Extract target items
        target_items = set()
        for item in target_order.get('items', target_order.get('target_order', [])):
            if isinstance(item, dict):
                name = item.get('name', '').lower()
                for word in name.split():
                    target_items.add(word)
        
        if not mentioned_items and not target_items:
            return 0.0
        
        # Calculate overlap
        overlap = len(mentioned_items & target_items)
        total = len(mentioned_items | target_items)
        
        return overlap / total if total > 0 else 0.0