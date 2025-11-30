"""
LLM evaluator prompts for extracting states from conversations without explicit tracking.
Used primarily for Experiment 1 (LLM baseline) and partially for Experiment 2.
"""

ORDER_TRACKING_EVALUATOR_PROMPT = """
Extract order state BEFORE guest's latest message.

Conversation:
{conversation_history}

Guest says: {guest_message}

What did the guest know BEFORE speaking?
- current_items_in_order: What restaurant confirmed so far
- target_items_to_order: What guest wants to order

Return JSON:
{{
  "current_items_in_order": [
    {{
      "name": "item name",
      "quantity": 1,
      "base_price": 0.0,
      "item_total_price": 0.0,
      "modifiers": [
        {{
          "modifier_type_name": "type (e.g., Bread Choice)",
          "modifier_option_name": "option (e.g., White)",
          "is_required": true,
          "price": 0.0
        }}
      ]
    }}
  ],
  "target_items_to_order": [
    {{
      "name": "item name",
      "quantity": 1,
      "modifiers": ["modifier1", "modifier2"]
    }}
  ]
}}

Note: For current_items_in_order, extract as much detail as available from the restaurant's confirmation. If prices aren't mentioned, use 0.0.
"""

BEHAVIORAL_STATE_EVALUATOR_PROMPT = """
Analyze the behavioral state that led to the guest's message.

Previous conversation context (what guest knows BEFORE speaking):
{previous_context}

Guest's message: {guest_message}

Based on the restaurant's PREVIOUS messages (not the guest's current message), classify:
- is_ordering_complete: Based on restaurant's previous responses, does the guest believe their order is complete? (yes/no)

Based on the guest's CURRENT message, classify:
- next_message_menu_exploration_style: Does THIS guest message explore menu or order directly? (explores_menu/does_not_explore_menu)
- next_message_mood_tone: What mood is shown in THIS guest message? (casual/frustrated/confused/enthusiastic)
- next_message_ordering_style: How does THIS guest message order items? (one_by_one/all_at_once)

Return JSON:
{{
  "is_ordering_complete": "yes/no",
  "next_message_menu_exploration_style": "explores_menu/does_not_explore_menu",
  "next_message_mood_tone": "casual/frustrated/confused/enthusiastic",
  "next_message_ordering_style": "one_by_one/all_at_once"
}}
"""

PERSONA_TRAITS_EVALUATOR_PROMPT = """
Given this guest message and persona, evaluate how well the message adheres to the persona traits.

Guest message: {guest_message}
Persona description: {persona}

Analyze the following traits:
1. Communication style (brevity, formality, efficiency)
2. Ordering approach (decisive vs exploratory)
3. Personality consistency

Return a score from 0.0 to 1.0 where:
- 1.0 = Perfect adherence to persona
- 0.5 = Partial adherence
- 0.0 = Contradicts persona

Return as JSON:
{{
  "score": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""