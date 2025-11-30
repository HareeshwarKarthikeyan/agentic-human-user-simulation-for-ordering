"""
Unified state extraction module for all experiment levels.
Handles both explicit state extraction (from agent tool calls) and 
inferred state extraction (using LLM evaluators).

Experiment Structure:
- Exp1: LLM baseline (no agents/tools)
- Exp2: Guest Agent only (basic tools: name, persona, target, history)
- Exp3: Guest Agent + Order Tracking Agent
- Exp4: Guest Agent + Message Attributes Agent  
- Exp5: Full system (Guest Agent + Order Tracking + Message Attributes)
"""

import json
from typing import Dict, List, Any, Optional
from .llm_evaluator_prompts import (
    ORDER_TRACKING_EVALUATOR_PROMPT,
    BEHAVIORAL_STATE_EVALUATOR_PROMPT
)


class UnifiedStateExtractor:
    """
    Extracts order tracking and behavioral states from conversation logs
    across all experiment levels.
    """
    
    def __init__(self, experiment_level: int, llm_client=None):
        """
        Initialize the state extractor.
        
        Args:
            experiment_level: 1-5 indicating the ablation level
                1: LLM baseline (no agents/tools)
                2: Guest Agent only (basic tools)
                3: Guest Agent + Order Tracking Agent
                4: Guest Agent + Message Attributes Agent
                5: Full system (all agents)
            llm_client: LLM client for inferring states
        """
        self.experiment_level = experiment_level
        self.llm_client = llm_client
        
        # Define which experiments need LLM for each type of state
        self.needs_llm_for_order = [1, 2, 4]  # No order tracking agent
        self.needs_llm_for_behavioral = [1, 2, 3]  # No message attributes agent
        
        # Check if LLM client is needed
        needs_llm = set(self.needs_llm_for_order + self.needs_llm_for_behavioral)
        if experiment_level in needs_llm and not llm_client:
            raise ValueError(f"LLM client required for experiment {experiment_level}")
    
    def extract_order_tracking_state(self, 
                                    conversation_log: Dict, 
                                    message_idx: int) -> Optional[Dict]:
        """
        Extract order tracking state at a specific message index.
        
        Args:
            conversation_log: Full conversation log
            message_idx: Index of the message to extract state for
            
        Returns:
            Order tracking state dict or None if not found
        """
        if self.experiment_level in [3, 5]:
            # Experiments 3 and 5 have order tracking agent
            return self._extract_order_state_from_tools(conversation_log, message_idx)
        else:
            # Experiments 1, 2, 4: Use LLM to infer (no order tracking agent)
            return self._infer_order_state_with_llm(conversation_log, message_idx)
    
    def extract_behavioral_state(self, 
                                conversation_log: Dict, 
                                message_idx: int, 
                                persona: str) -> Optional[Dict]:
        """
        Extract behavioral state at a specific message index.
        
        Args:
            conversation_log: Full conversation log
            message_idx: Index of the message
            persona: Persona description for context
            
        Returns:
            Behavioral state dict or None if not found
        """
        if self.experiment_level in [4, 5]:
            # Experiments 4 and 5 have message attributes agent
            return self._extract_behavioral_state_from_tools(conversation_log, message_idx)
        else:
            # Experiments 1, 2, 3: Use LLM to infer (no message attributes agent)
            return self._infer_behavioral_state_with_llm(conversation_log, message_idx, persona)
    
    def _extract_order_state_from_tools(self, log: Dict, idx: int) -> Optional[Dict]:
        """Extract order state from agent tool calls."""
        message = log['messages'][idx] if 'messages' in log else log[idx]
        tool_calls = message.get('tool_calls', {})
        
        # Handle string representation of tool_calls
        if isinstance(tool_calls, str) and tool_calls:
            # Fix escape sequences in the tool_calls string (for exp3 and exp5)
            # This handles raw log files that haven't been pre-processed
            tool_calls = tool_calls.replace("\\'", "'").replace('\\"', '"')
            # Parse the OrderTracking result from the string
            import re
            import ast
            
            # Look for update_order_tracking tool call result
            if 'update_order_tracking' in tool_calls:
                # The OrderTracking is embedded in the 'result' field of the tool call
                # Look for 'result': followed by the value (can be single or double quotes)
                # Pattern 1: Single quotes around the value
                pattern1 = r"'result':\s*'(.*?)',\s*'(?:decision_latency|execution_latency)"
                matches1 = re.findall(pattern1, tool_calls)
                
                # Pattern 2: Double quotes around the value
                pattern2 = r"'result':\s*\"(.*?)\",\s*'(?:decision_latency|execution_latency)"
                matches2 = re.findall(pattern2, tool_calls)
                
                # Combine all matches
                result_matches = matches1 + matches2
                
                # Filter to only matches containing order tracking
                result_matches = [r for r in result_matches if 'Latest Order tracking' in r]
                
                # Get the LAST (most recent) order tracking state
                last_order_state = None
                
                for result_str in result_matches:
                    # Check if this result contains order tracking info
                    if 'current_items_in_order' in result_str and 'target_items_to_order' in result_str:
                        # Extract the actual lists from the result string
                        # The format is like: "Latest Order tracking : current_items_in_order=[...] target_items_to_order=[...]"
                        
                        # Find the OrderTracking data section
                        current_start = result_str.find('current_items_in_order=')
                        target_start = result_str.find('target_items_to_order=')
                        
                        if current_start >= 0 and target_start >= 0:
                            # Extract current items list
                            current_start += len('current_items_in_order=')
                            current_end = target_start - 1  # Before 'target_items_to_order'
                            current_str = result_str[current_start:current_end].strip()
                            
                            # Extract target items list
                            target_start += len('target_items_to_order=')
                            target_str = result_str[target_start:].strip()
                            
                            # Parse the lists
                            current_items = self._parse_order_items_from_string(current_str)
                            target_items = self._parse_order_items_from_string(target_str)
                            
                            # Store this as the latest state (will be overwritten by subsequent matches)
                            last_order_state = {
                                'current_items_in_order': current_items,
                                'target_items_to_order': target_items
                            }
                
                # Return the last (most recent) order state found
                if last_order_state:
                    return last_order_state
            
            return None
        
        # Search through nested tool call structure (when tool_calls is a dict)
        for agent_name, agent_data in tool_calls.items():
            if 'tool_calls' in agent_data:
                for call_id, call_data in agent_data['tool_calls'].items():
                    tool_name = call_data.get('tool_name', '').lower()
                    if 'order_tracking' in tool_name:
                        result = call_data.get('result')
                        if isinstance(result, str):
                            # Parse if it's a string representation of OrderTracking
                            if 'OrderTracking(' in result:
                                # Use the same parsing logic as above
                                return self._parse_order_tracking_string(result)
                            else:
                                try:
                                    return json.loads(result)
                                except:
                                    pass
                        elif isinstance(result, dict):
                            return result
        
        # Also check for sub_agents
        if 'sub_agents' in tool_calls:
            for sub_agent_data in tool_calls['sub_agents']:
                if 'tool_calls' in sub_agent_data:
                    for call_id, call_data in sub_agent_data['tool_calls'].items():
                        if 'order_tracking' in call_data.get('tool_name', '').lower():
                            result = call_data.get('result')
                            if isinstance(result, str) and 'OrderTracking(' in result:
                                return self._parse_order_tracking_string(result)
                            return result
        
        return None
    
    def _parse_order_items_from_string(self, items_str):
        """Parse list of OrderItem objects from string."""
        import re
        items = []
        
        # Handle empty list
        if items_str.strip() == '[]':
            return items
        
        idx = 0
        while idx < len(items_str):
            if items_str[idx:].startswith('OrderItem('):
                start = idx
                depth = 1
                idx += len('OrderItem(')
                while idx < len(items_str) and depth > 0:
                    if items_str[idx] == '(':
                        depth += 1
                    elif items_str[idx] == ')':
                        depth -= 1
                    idx += 1
                item_str = items_str[start:idx]
                
                # Parse individual OrderItem
                item_dict = {}
                # Extract fields
                name_match = re.search(r"name='([^']*)'", item_str)
                if name_match:
                    item_dict['name'] = name_match.group(1)
                
                quantity_match = re.search(r'quantity=(\d+)', item_str)
                if quantity_match:
                    item_dict['quantity'] = int(quantity_match.group(1))
                
                base_price_match = re.search(r'base_price=([\d.]+)', item_str)
                if base_price_match:
                    item_dict['base_price'] = float(base_price_match.group(1))
                
                total_price_match = re.search(r'item_total_price=([\d.]+)', item_str)
                if total_price_match:
                    item_dict['item_total_price'] = float(total_price_match.group(1))
                
                # Extract modifiers list
                modifiers_match = re.search(r'modifiers=(\[.*?\])', item_str)
                if modifiers_match:
                    modifiers_str = modifiers_match.group(1)
                    modifiers = []
                    
                    # Parse each Modifier object
                    mod_idx = 0
                    while mod_idx < len(modifiers_str):
                        if modifiers_str[mod_idx:].startswith('Modifier('):
                            mod_start = mod_idx
                            mod_depth = 1
                            mod_idx += len('Modifier(')
                            while mod_idx < len(modifiers_str) and mod_depth > 0:
                                if modifiers_str[mod_idx] == '(':
                                    mod_depth += 1
                                elif modifiers_str[mod_idx] == ')':
                                    mod_depth -= 1
                                mod_idx += 1
                            mod_str = modifiers_str[mod_start:mod_idx]
                            
                            # Parse modifier fields
                            mod_dict = {}
                            type_match = re.search(r"modifier_type_name='([^']*)'", mod_str)
                            if type_match:
                                mod_dict['modifier_type_name'] = type_match.group(1)
                            
                            option_match = re.search(r"modifier_option_name='([^']*)'", mod_str)
                            if option_match:
                                mod_dict['modifier_option_name'] = option_match.group(1)
                            
                            required_match = re.search(r'is_required=(True|False)', mod_str)
                            if required_match:
                                mod_dict['is_required'] = required_match.group(1) == 'True'
                            
                            price_match = re.search(r'price=([\d.]+)', mod_str)
                            if price_match:
                                mod_dict['price'] = float(price_match.group(1))
                            
                            if mod_dict:
                                modifiers.append(mod_dict)
                        else:
                            mod_idx += 1
                    
                    item_dict['modifiers'] = modifiers
                else:
                    item_dict['modifiers'] = []
                
                if item_dict and 'name' in item_dict:
                    items.append(item_dict)
            else:
                idx += 1
        
        return items
    
    def _parse_order_tracking_string(self, order_tracking_str):
        """Helper method to parse OrderTracking string representation."""
        import re
        
        def parse_order_items(items_str):
            """Parse list of OrderItem objects from string."""
            items = []
            idx = 0
            while idx < len(items_str):
                if items_str[idx:].startswith('OrderItem('):
                    start = idx
                    depth = 1
                    idx += len('OrderItem(')
                    while idx < len(items_str) and depth > 0:
                        if items_str[idx] == '(':
                            depth += 1
                        elif items_str[idx] == ')':
                            depth -= 1
                        idx += 1
                    item_str = items_str[start:idx]
                    
                    # Parse individual OrderItem
                    item_dict = {}
                    # Extract fields
                    name_match = re.search(r"name='([^']*)'", item_str)
                    if name_match:
                        item_dict['name'] = name_match.group(1)
                    
                    quantity_match = re.search(r'quantity=(\d+)', item_str)
                    if quantity_match:
                        item_dict['quantity'] = int(quantity_match.group(1))
                    
                    base_price_match = re.search(r'base_price=([\d.]+)', item_str)
                    if base_price_match:
                        item_dict['base_price'] = float(base_price_match.group(1))
                    
                    total_price_match = re.search(r'item_total_price=([\d.]+)', item_str)
                    if total_price_match:
                        item_dict['item_total_price'] = float(total_price_match.group(1))
                    
                    # Extract modifiers list
                    modifiers_match = re.search(r'modifiers=(\[.*?\])', item_str)
                    if modifiers_match:
                        modifiers_str = modifiers_match.group(1)
                        modifiers = []
                        
                        # Parse each Modifier object
                        mod_idx = 0
                        while mod_idx < len(modifiers_str):
                            if modifiers_str[mod_idx:].startswith('Modifier('):
                                mod_start = mod_idx
                                mod_depth = 1
                                mod_idx += len('Modifier(')
                                while mod_idx < len(modifiers_str) and mod_depth > 0:
                                    if modifiers_str[mod_idx] == '(':
                                        mod_depth += 1
                                    elif modifiers_str[mod_idx] == ')':
                                        mod_depth -= 1
                                    mod_idx += 1
                                mod_str = modifiers_str[mod_start:mod_idx]
                                
                                # Parse modifier fields
                                mod_dict = {}
                                type_match = re.search(r"modifier_type_name='([^']*)'", mod_str)
                                if type_match:
                                    mod_dict['modifier_type_name'] = type_match.group(1)
                                
                                option_match = re.search(r"modifier_option_name='([^']*)'", mod_str)
                                if option_match:
                                    mod_dict['modifier_option_name'] = option_match.group(1)
                                
                                required_match = re.search(r'is_required=(True|False)', mod_str)
                                if required_match:
                                    mod_dict['is_required'] = required_match.group(1) == 'True'
                                
                                price_match = re.search(r'price=([\d.]+)', mod_str)
                                if price_match:
                                    mod_dict['price'] = float(price_match.group(1))
                                
                                if mod_dict:
                                    modifiers.append(mod_dict)
                            else:
                                mod_idx += 1
                        
                        item_dict['modifiers'] = modifiers
                    else:
                        item_dict['modifiers'] = []
                    
                    if item_dict and 'name' in item_dict:
                        items.append(item_dict)
                else:
                    idx += 1
            
            return items
        
        # Extract current and target items
        current_match = re.search(r'current_items_in_order=(\[.*?\])\s*,\s*target_items_to_order', order_tracking_str)
        target_match = re.search(r'target_items_to_order=(\[.*?\])', order_tracking_str)
        
        if current_match and target_match:
            current_items = parse_order_items(current_match.group(1))
            target_items = parse_order_items(target_match.group(1))
            
            return {
                'current_items_in_order': current_items,
                'target_items_to_order': target_items
            }
        
        return None
    
    def _extract_behavioral_state_from_tools(self, log: Dict, idx: int) -> Optional[Dict]:
        """Extract behavioral state from agent tool calls."""
        message = log['messages'][idx] if 'messages' in log else log[idx]
        tool_calls = message.get('tool_calls', {})
        
        # Handle string representation of tool_calls
        if isinstance(tool_calls, str) and tool_calls:
            # Parse the MessageGenerationAttributes from the string
            import re
            
            # For Exp4: Look for get_message_generation_attributes_for_next_message tool call
            if 'get_message_generation_attributes_for_next_message' in tool_calls:
                # Extract the result of this specific tool call
                pattern = r"'tool_name': 'get_message_generation_attributes_for_next_message'.*?'result': \"([^\"]+)\""
                match = re.search(pattern, tool_calls, re.DOTALL)
                if match:
                    result_str = match.group(1)
                    attributes = {}
                    
                    # Parse the attributes from the result string
                    patterns = {
                        'is_ordering_complete': r'is_ordering_complete=<IsOrderingComplete\.(\w+):',
                        'next_message_menu_exploration_style': r'next_message_menu_exploration_style=<MenuExplorationStyle\.(\w+):',
                        'next_message_mood_tone': r'next_message_mood_tone=<MessageMoodTone\.(\w+):',
                        'next_message_ordering_style': r'next_message_ordering_style=<OrderingStyle\.(\w+):'
                    }
                    
                    for attr, pattern in patterns.items():
                        attr_match = re.search(pattern, result_str)
                        if attr_match:
                            value = attr_match.group(1).lower()
                            if attr == 'is_ordering_complete':
                                attributes[attr] = 'yes' if value == 'yes' else 'no'
                            elif 'exploration' in attr:
                                # Handle both DOES_NOT_EXPLORE_MENU and EXPLORES_MENU
                                if 'does_not_explore' in value or 'not_explore' in value:
                                    attributes[attr] = 'does_not_explore_menu'
                                else:
                                    attributes[attr] = 'explores_menu'
                            elif 'ordering_style' in attr:
                                # Handle ONE_BY_ONE and ALL_AT_ONCE
                                if 'one_by_one' in value or 'one' in value and 'by' in value:
                                    attributes[attr] = 'one_by_one'
                                else:
                                    attributes[attr] = 'all_at_once'
                            else:
                                # For mood tone, just use the value as is
                                attributes[attr] = value
                    
                    return attributes if attributes else None
            
            # For Exp5: Check if attributes are present directly (either in MessageGenerationAttributes or directly)
            elif 'is_ordering_complete' in tool_calls:
                attributes = {}
                
                patterns = {
                    'is_ordering_complete': r'is_ordering_complete=<IsOrderingComplete\.(\w+):',
                    'next_message_menu_exploration_style': r'next_message_menu_exploration_style=<MenuExplorationStyle\.(\w+):',
                    'next_message_mood_tone': r'next_message_mood_tone=<MessageMoodTone\.(\w+):',
                    'next_message_ordering_style': r'next_message_ordering_style=<OrderingStyle\.(\w+):'
                }
                
                for attr, pattern in patterns.items():
                    match = re.search(pattern, tool_calls)
                    if match:
                        value = match.group(1).lower()
                        if attr == 'is_ordering_complete':
                            attributes[attr] = 'yes' if value == 'yes' else 'no'
                        elif 'exploration' in attr:
                            attributes[attr] = 'explores_menu' if 'explore' in value else 'does_not_explore_menu'
                        elif 'ordering_style' in attr:
                            attributes[attr] = 'one_by_one' if 'one' in value else 'all_at_once'
                        else:
                            attributes[attr] = value
                
                return attributes if attributes else None
            return None
        
        # Search through nested tool call structure
        for agent_name, agent_data in tool_calls.items():
            if 'tool_calls' in agent_data:
                for call_id, call_data in agent_data['tool_calls'].items():
                    tool_name = call_data.get('tool_name', '').lower()
                    if 'message_generation_attributes' in tool_name or 'message_attributes' in tool_name:
                        result = call_data.get('result')
                        if isinstance(result, str):
                            # Parse MessageGenerationAttributes object
                            import re
                            attributes = {}
                            
                            # Extract each attribute
                            patterns = {
                                'is_ordering_complete': r'is_ordering_complete=<IsOrderingComplete\.(\w+):',
                                'next_message_menu_exploration_style': r'next_message_menu_exploration_style=<MenuExplorationStyle\.(\w+):',
                                'next_message_mood_tone': r'next_message_mood_tone=<MessageMoodTone\.(\w+):',
                                'next_message_ordering_style': r'next_message_ordering_style=<OrderingStyle\.(\w+):'
                            }
                            
                            for attr, pattern in patterns.items():
                                match = re.search(pattern, result)
                                if match:
                                    value = match.group(1).lower()
                                    if attr == 'is_ordering_complete':
                                        attributes[attr] = 'yes' if value == 'yes' else 'no'
                                    elif 'exploration' in attr:
                                        attributes[attr] = 'explores_menu' if 'explore' in value else 'does_not_explore_menu'
                                    elif 'ordering_style' in attr:
                                        attributes[attr] = 'one_by_one' if 'one' in value else 'all_at_once'
                                    else:
                                        attributes[attr] = value
                            
                            return attributes if attributes else None
                        elif isinstance(result, dict):
                            return result
        
        # Also check for sub_agents
        if 'sub_agents' in tool_calls:
            for sub_agent_data in tool_calls['sub_agents']:
                if 'tool_calls' in sub_agent_data:
                    for call_id, call_data in sub_agent_data['tool_calls'].items():
                        tool_name = call_data.get('tool_name', '').lower()
                        if 'message_generation_attributes' in tool_name or 'message_attributes' in tool_name:
                            return self._parse_behavioral_result(call_data.get('result'))
        
        return None
    
    def _parse_behavioral_result(self, result):
        """Helper to parse behavioral state result."""
        if isinstance(result, str):
            # Parse MessageGenerationAttributes object
            import re
            attributes = {}
            
            patterns = {
                'is_ordering_complete': r'is_ordering_complete=<IsOrderingComplete\.(\w+):',
                'next_message_menu_exploration_style': r'next_message_menu_exploration_style=<MenuExplorationStyle\.(\w+):',
                'next_message_mood_tone': r'next_message_mood_tone=<MessageMoodTone\.(\w+):',
                'next_message_ordering_style': r'next_message_ordering_style=<OrderingStyle\.(\w+):'
            }
            
            for attr, pattern in patterns.items():
                match = re.search(pattern, result)
                if match:
                    value = match.group(1).lower()
                    if attr == 'is_ordering_complete':
                        attributes[attr] = 'yes' if value == 'yes' else 'no'
                    elif 'exploration' in attr:
                        attributes[attr] = 'explores_menu' if 'explore' in value else 'does_not_explore_menu'
                    elif 'ordering_style' in attr:
                        attributes[attr] = 'one_by_one' if 'one' in value else 'all_at_once'
                    else:
                        attributes[attr] = value
            
            return attributes if attributes else None
        elif isinstance(result, dict):
            return result
        return None
    
    def _infer_order_state_with_llm(self, log: Dict, idx: int) -> Dict:
        """Use LLM to infer order state from conversation."""
        conversation_history = self._build_conversation_history(log, idx)
        messages = log['messages'] if 'messages' in log else log
        guest_message = messages[idx]['content']
        
        prompt = ORDER_TRACKING_EVALUATOR_PROMPT.format(
            conversation_history=conversation_history,
            guest_message=guest_message
        )
        
        response = self.llm_client.complete(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Return empty state if parsing fails
            return {
                'current_items_in_order': [],
                'target_items_to_order': []
            }
    
    def _infer_behavioral_state_with_llm(self, log: Dict, idx: int, persona: str) -> Dict:
        """Use LLM to infer behavioral state from conversation."""
        messages = log['messages'] if 'messages' in log else log
        guest_message = messages[idx]['content']
        previous_context = self._get_previous_context(log, idx)
        
        prompt = BEHAVIORAL_STATE_EVALUATOR_PROMPT.format(
            guest_message=guest_message,
            previous_context=previous_context,
            persona=persona
        )
        
        response = self.llm_client.complete(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Return default state if parsing fails
            return {
                'is_ordering_complete': 'no',
                'next_message_menu_exploration_style': 'does_not_explore_menu',
                'next_message_mood_tone': 'casual',
                'next_message_ordering_style': 'all_at_once'
            }
    
    def _build_conversation_history(self, log: Dict, up_to_idx: int) -> str:
        """Build conversation history up to given index - optimized to include only essential context."""
        messages = log['messages'] if 'messages' in log else log
        history = []
        
        # For order tracking: Include all messages (need to track confirmations)
        # But limit to last 10 messages for very long conversations
        start_idx = max(0, up_to_idx - 9) if up_to_idx > 10 else 0
        
        for i in range(start_idx, min(up_to_idx + 1, len(messages))):
            msg = messages[i]
            role = "Guest" if msg.get('role') in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] else "Restaurant"
            content = msg.get('content', '')
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            history.append(f"{role}: {content}")
        
        return "\n".join(history)
    
    def _get_previous_context(self, log: Dict, idx: int) -> str:
        """Get previous messages for context, especially the last restaurant message."""
        messages = log['messages'] if 'messages' in log else log
        context = []
        
        # Get up to 3 previous messages, ensuring we include the last restaurant message
        for i in range(max(0, idx-3), idx):
            if i < len(messages):
                msg = messages[i]
                role = "Guest" if msg.get('role') in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm'] else "Restaurant"
                content = msg.get('content', '')
                # Truncate very long messages
                if len(content) > 300:
                    content = content[:300] + "..."
                context.append(f"{role}: {content}")
        
        # If no context, explicitly state it
        if not context:
            return "No previous messages (this is the first guest message)"
        
        return "\n".join(context)