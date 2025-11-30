"""
Final state extractor - extracts order state only once per conversation at the last guest message.
This is much faster than extracting state for every message.
"""

import json
from typing import Dict, List, Optional, Any


FINAL_ORDER_STATE_PROMPT = """
Given this FULL conversation, extract the final order state at the end.

Target items the guest wants to order (for reference):
{target_items}

Full Conversation (from start to end):
{conversation_history}

Based on the ENTIRE conversation above, extract:
- current_items_in_order: What items has the restaurant explicitly confirmed/added to the order by the end?
- target_items_to_order: Use the target items provided above

Look through the WHOLE conversation for restaurant confirmations like:
- "I've added X to your order"
- "Your order includes X"
- "That's X for $Y"
- Any explicit confirmation of items

Return ONLY valid JSON:
{{
  "current_items_in_order": [
    {{"name": "item", "quantity": 1, "base_price": 0.0, "item_total_price": 0.0, "modifiers": []}}
  ],
  "target_items_to_order": [
    {{"name": "item", "quantity": 1, "modifiers": []}}
  ]
}}
"""


class FinalStateExtractor:
    """
    Extracts order state only once per conversation - at the final guest message.
    Much faster than per-message extraction.
    """
    
    def __init__(self, experiment_level: int, llm_client):
        self.experiment_level = experiment_level
        self.llm_client = llm_client
        
        # Define which experiments need LLM for order tracking
        self.needs_llm_for_order = experiment_level in [1, 2, 4]
        
    def extract_final_order_state(self, conversation_log: Dict, target_order: Dict = None) -> Dict:
        """
        Extract the final order state from the FULL conversation.
        
        Args:
            conversation_log: Full conversation log
            target_order: Target order specification (what guest wants to order)
            
        Returns:
            Final order state dict
        """
        messages = conversation_log.get('messages', [])
        
        # Build FULL conversation from start to end
        conversation_parts = []
        has_guest_messages = False
        
        for idx, msg in enumerate(messages):
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # Skip system messages and exit
            if role == 'system' or content == 'exit':
                continue
                
            is_guest = role in ['agent', 'guest', 'guest_simulation_agent', 'guest_simulation_llm']
            
            if is_guest:
                has_guest_messages = True
                conversation_parts.append(f"[Guest] {content}")
            else:
                conversation_parts.append(f"[Restaurant] {content}")
        
        if not has_guest_messages:
            # No guest messages found
            return {
                'current_items_in_order': [],
                'target_items_to_order': []
            }
        
        # Format target items for reference
        target_items_str = ""
        if target_order:
            items = []
            for item in target_order.get('order_items', []):
                item_str = f"- {item.get('name', '')}"
                if item.get('quantity', 1) > 1:
                    item_str += f" x{item.get('quantity')}"
                modifiers = []
                for mod in item.get('modifiers', []):
                    modifiers.append(mod.get('modifier_option_name', ''))
                if modifiers:
                    item_str += f" with {', '.join(modifiers)}"
                items.append(item_str)
            target_items_str = "\n".join(items)
        else:
            target_items_str = "Not specified - extract from conversation"
        
        # Build full conversation history
        conversation_history = "\n".join(conversation_parts)
        
        # Make single LLM call for final state with FULL conversation
        prompt = FINAL_ORDER_STATE_PROMPT.format(
            target_items=target_items_str,
            conversation_history=conversation_history
        )
        
        response = self.llm_client.complete(prompt)
        
        try:
            result = json.loads(response)
            
            # If we have target_order, use it for target_items_to_order
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
            else:
                target_items = result.get('target_items_to_order', [])
            
            return {
                'current_items_in_order': result.get('current_items_in_order', []),
                'target_items_to_order': target_items
            }
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                result = json.loads(response.strip())
                
                # Use target_order for target_items if available
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
                else:
                    target_items = result.get('target_items_to_order', [])
                
                return {
                    'current_items_in_order': result.get('current_items_in_order', []),
                    'target_items_to_order': target_items
                }
            except:
                # Return empty current items but use target_order if available
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
                else:
                    target_items = []
                    
                return {
                    'current_items_in_order': [],
                    'target_items_to_order': target_items
                }