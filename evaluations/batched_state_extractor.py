"""
Batched state extraction to minimize LLM calls.
Makes only 2 LLM calls per conversation instead of 2N calls.
"""

import json
from typing import Dict, List, Optional, Any


BATCH_ORDER_TRACKING_PROMPT = """
Extract the order state BEFORE each guest message (i.e., what the guest knows from the restaurant's previous messages).

Conversation:
{conversation_history}

For each [Guest Message X], extract the state based on what happened BEFORE the guest speaks:
- current_items_in_order: What items has the restaurant confirmed BEFORE this guest message?
- target_items_to_order: What items does the guest want to order (constant throughout)?

Return ONLY valid JSON:
{{
  "states": [
    {{
      "message_number": 1,
      "current_items_in_order": [],
      "target_items_to_order": [{{"name": "Burger", "quantity": 1, "modifiers": []}}]
    }},
    {{
      "message_number": 2,
      "current_items_in_order": [{{"name": "Burger", "quantity": 1, "base_price": 12.99, "item_total_price": 12.99, "modifiers": []}}],
      "target_items_to_order": [{{"name": "Burger", "quantity": 1, "modifiers": []}}]
    }}
  ]
}}"""

BATCH_BEHAVIORAL_PROMPT = """
Extract the behavioral state that led to each guest message.

Conversation:
{conversation_history}

Persona: {persona}

For each [Guest Message X], determine:

BASED ON RESTAURANT'S MESSAGES BEFORE THIS GUEST MESSAGE:
- is_ordering_complete: Looking at what the restaurant said BEFORE this guest message, does the guest believe their order is complete? (yes/no)

BASED ON THIS GUEST MESSAGE ITSELF:
- next_message_menu_exploration_style: Does THIS guest message explore menu or order directly?
- next_message_mood_tone: What mood is shown IN THIS guest message?
- next_message_ordering_style: How does THIS guest message order items?

Return ONLY valid JSON:
{{
  "states": [
    {{
      "message_number": 1,
      "is_ordering_complete": "no",
      "next_message_menu_exploration_style": "does_not_explore_menu",
      "next_message_mood_tone": "casual",
      "next_message_ordering_style": "all_at_once"
    }}
  ]
}}"""


class BatchedStateExtractor:
    """
    Optimized state extractor that batches LLM calls.
    Extracts all states for a conversation in 2 LLM calls instead of 2N calls.
    """
    
    def __init__(self, experiment_level: int, llm_client):
        self.experiment_level = experiment_level
        self.llm_client = llm_client
        
        # Define which experiments need LLM for each type
        self.needs_llm_for_order = experiment_level in [1, 2, 4]
        self.needs_llm_for_behavioral = experiment_level in [1, 2, 3]
    
    def extract_all_states_batched(self, 
                                   conversation_log: Dict,
                                   persona: str = "") -> Dict[int, Dict]:
        """
        Extract all states for a conversation with minimal LLM calls.
        
        Args:
            conversation_log: Full conversation log
            persona: Guest persona description
            
        Returns:
            Dictionary mapping message index to extracted states
        """
        messages = conversation_log.get('messages', [])
        
        # Find guest message indices and build marked conversation
        guest_indices = []
        guest_message_numbers = {}  # Map index to message number
        conversation_parts = []
        guest_count = 0
        
        for idx, msg in enumerate(messages):
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # Check if guest message
            is_guest = role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm']
            
            if is_guest and content != 'exit':
                guest_count += 1
                guest_indices.append(idx)
                guest_message_numbers[idx] = guest_count
                conversation_parts.append(f"[Guest Message {guest_count}] {content}")
            elif is_guest and content == 'exit':
                conversation_parts.append(f"[Guest Exit] {content}")
            else:
                # Restaurant message
                conversation_parts.append(f"[Restaurant] {content}")
        
        if not guest_indices:
            return {}
        
        conversation_history = "\n".join(conversation_parts)
        
        # Initialize result
        states_by_index = {}
        
        # Batch extract order states if needed (1 LLM call)
        if self.needs_llm_for_order:
            order_states = self._batch_extract_order_states(
                conversation_history, 
                guest_indices, 
                guest_message_numbers
            )
            for idx in guest_indices:
                if idx not in states_by_index:
                    states_by_index[idx] = {}
                states_by_index[idx]['order_state'] = order_states.get(idx, {
                    'current_items_in_order': [],
                    'target_items_to_order': []
                })
        
        # Batch extract behavioral states if needed (1 LLM call)
        if self.needs_llm_for_behavioral:
            behavioral_states = self._batch_extract_behavioral_states(
                conversation_history,
                guest_indices,
                guest_message_numbers,
                persona
            )
            for idx in guest_indices:
                if idx not in states_by_index:
                    states_by_index[idx] = {}
                states_by_index[idx]['behavioral_state'] = behavioral_states.get(idx, {
                    'is_ordering_complete': 'no',
                    'next_message_menu_exploration_style': 'does_not_explore_menu',
                    'next_message_mood_tone': 'casual',
                    'next_message_ordering_style': 'all_at_once'
                })
        
        return states_by_index
    
    def _batch_extract_order_states(self, 
                                   conversation_history: str,
                                   guest_indices: List[int],
                                   guest_message_numbers: Dict[int, int]) -> Dict[int, Dict]:
        """Extract order states for all guest messages in one LLM call."""
        prompt = BATCH_ORDER_TRACKING_PROMPT.format(
            conversation_history=conversation_history
        )
        
        response = self.llm_client.complete(prompt)
        
        try:
            # Try to parse the response
            result_json = json.loads(response)
            states_list = result_json.get('states', [])
            
            # Map message numbers back to indices
            states_by_index = {}
            for idx in guest_indices:
                msg_num = guest_message_numbers[idx]
                # Find the state for this message number
                for state in states_list:
                    if state.get('message_number') == msg_num:
                        states_by_index[idx] = {
                            'current_items_in_order': state.get('current_items_in_order', []),
                            'target_items_to_order': state.get('target_items_to_order', [])
                        }
                        break
                
                # Default if not found
                if idx not in states_by_index:
                    states_by_index[idx] = {
                        'current_items_in_order': [],
                        'target_items_to_order': []
                    }
            
            return states_by_index
            
        except (json.JSONDecodeError, KeyError) as e:
            # Try to fix common JSON issues
            try:
                # Remove any markdown formatting
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                # Try parsing again
                result_json = json.loads(response.strip())
                states_list = result_json.get('states', [])
                
                # Map message numbers back to indices  
                states_by_index = {}
                for idx in guest_indices:
                    msg_num = guest_message_numbers[idx]
                    for state in states_list:
                        if state.get('message_number') == msg_num:
                            states_by_index[idx] = {
                                'current_items_in_order': state.get('current_items_in_order', []),
                                'target_items_to_order': state.get('target_items_to_order', [])
                            }
                            break
                    if idx not in states_by_index:
                        states_by_index[idx] = {'current_items_in_order': [], 'target_items_to_order': []}
                
                return states_by_index
            except:
                # If still fails, return empty states
                return {idx: {'current_items_in_order': [], 'target_items_to_order': []} 
                       for idx in guest_indices}
    
    def _batch_extract_behavioral_states(self,
                                        conversation_history: str,
                                        guest_indices: List[int],
                                        guest_message_numbers: Dict[int, int],
                                        persona: str) -> Dict[int, Dict]:
        """Extract behavioral states for all guest messages in one LLM call."""
        prompt = BATCH_BEHAVIORAL_PROMPT.format(
            conversation_history=conversation_history,
            persona=persona
        )
        
        response = self.llm_client.complete(prompt)
        
        try:
            # Try to parse the response
            result_json = json.loads(response)
            states_list = result_json.get('states', [])
            
            # Map message numbers back to indices
            states_by_index = {}
            for idx in guest_indices:
                msg_num = guest_message_numbers[idx]
                # Find the state for this message number
                for state in states_list:
                    if state.get('message_number') == msg_num:
                        states_by_index[idx] = {
                            'is_ordering_complete': state.get('is_ordering_complete', 'no'),
                            'next_message_menu_exploration_style': state.get('next_message_menu_exploration_style', 'does_not_explore_menu'),
                            'next_message_mood_tone': state.get('next_message_mood_tone', 'casual'),
                            'next_message_ordering_style': state.get('next_message_ordering_style', 'all_at_once')
                        }
                        break
                
                # Default if not found
                if idx not in states_by_index:
                    states_by_index[idx] = {
                        'is_ordering_complete': 'no',
                        'next_message_menu_exploration_style': 'does_not_explore_menu',
                        'next_message_mood_tone': 'casual',
                        'next_message_ordering_style': 'all_at_once'
                    }
            
            return states_by_index
            
        except (json.JSONDecodeError, KeyError) as e:
            # Try to fix common JSON issues
            try:
                # Remove any markdown formatting
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                # Try parsing again
                result_json = json.loads(response.strip())
                states_list = result_json.get('states', [])
                
                # Map message numbers back to indices
                states_by_index = {}
                for idx in guest_indices:
                    msg_num = guest_message_numbers[idx]
                    for state in states_list:
                        if state.get('message_number') == msg_num:
                            states_by_index[idx] = {
                                'is_ordering_complete': state.get('is_ordering_complete', 'no'),
                                'next_message_menu_exploration_style': state.get('next_message_menu_exploration_style', 'does_not_explore_menu'),
                                'next_message_mood_tone': state.get('next_message_mood_tone', 'casual'),
                                'next_message_ordering_style': state.get('next_message_ordering_style', 'all_at_once')
                            }
                            break
                    if idx not in states_by_index:
                        states_by_index[idx] = {
                            'is_ordering_complete': 'no',
                            'next_message_menu_exploration_style': 'does_not_explore_menu',
                            'next_message_mood_tone': 'casual',
                            'next_message_ordering_style': 'all_at_once'
                        }
                
                return states_by_index
            except:
                # If still fails, return default states
                return {idx: {
                    'is_ordering_complete': 'no',
                    'next_message_menu_exploration_style': 'does_not_explore_menu',
                    'next_message_mood_tone': 'casual',
                    'next_message_ordering_style': 'all_at_once'
                } for idx in guest_indices}