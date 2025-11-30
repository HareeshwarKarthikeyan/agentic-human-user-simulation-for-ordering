'''
We are running exp1 which is using a single LLM with all the guest info, target order info and past messages in the conversation so far fed to it. 
This experiment does not involve any agentic systems.
'''
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path so we can import from other parent directories
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


from agents.order_tracking_agent import OrderTracking
from validation_llm_client.message import Message
from agents.utils import (
    get_latencies_for_pydantic_agent_interaction,
    run_agent)
from scripts.conversation_config_and_helpers import (
    load_test_case_for_conversation,
    setup_order_taking_llm_handler,
    get_order_taking_llm_response,
    setup_validation_llm_handler,
    get_validation_llm_response,
    VALIDATION_LLM_ROLE_NAME,
    ORDER_TAKING_LLM_ROLE_NAME,
    USER_ROLE_NAME,
    NUM_TEST_CASES,
    NUM_RUNS_PER_TEST_CASE,
)

def load_menu_string(menu_path: str):
    """Load menu from menu.json file and return as JSON string"""
    with open(menu_path, 'r') as f:
        menu_data = json.load(f)
        return json.dumps(menu_data, indent=2)


class ConversationClient:
    def __init__(self):
        self.chat_history = []
        self.run_iteration_number = 0
        self.test_case_index = 0
        self.test_case_name = ""
    
    def get_chat_messages_log(self) -> pd.DataFrame:
        """Get chat messages as a pandas DataFrame with all Message class fields as columns"""
        if not self.chat_history:
            print("No chat history available")
            return pd.DataFrame()
            
        # Prepare data for DataFrame
        data = []
        for message in self.chat_history:
            # Convert Message object to dictionary with all fields
            row = {
                'message_id': str(message.message_id),
                'role': message.role,
                'content': message.content,
                'name': message.name if message.name else '',
                'timestamp': message.timestamp.isoformat(),
                'tool_calls': str(message.tool_calls) if message.tool_calls else '',
                'logs': json.dumps(message.logs) if message.logs else '',
                'trace_id': message.trace_id if message.trace_id else '',
                'parent_id': str(message.parent_id) if message.parent_id else '',
                'usage': json.dumps(message.usage) if message.usage else '',
                'latency_ms': message.latency_ms if message.latency_ms else -1
            }
            data.append(row)
        
        # Create DataFrame with ordered columns
        columns = [
            'message_id',
            'role', 
            'content',
            'name',
            'timestamp',
            'tool_calls',
            'logs',
            'trace_id',
            'parent_id',
            'usage',
            'latency_ms'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        print(f"Retrieved {len(self.chat_history)} messages as DataFrame")
        return df


    async def run(self):

        menu_string = load_menu_string("../../test_case_generation/data/menu.json")
        order_taking_llm_handler = setup_order_taking_llm_handler(
            provider="openai",
            llm_role_name=ORDER_TAKING_LLM_ROLE_NAME,
            menu_string=menu_string,
        )

        test_case, test_case_name = load_test_case_for_conversation(
            test_case_index=test_case_index
        )
        self.test_case_name = test_case_name
        guest_llm_handler = setup_validation_llm_handler(
            provider="openai",
            llm_role_name=VALIDATION_LLM_ROLE_NAME + "-" + test_case_name,
            test_case=test_case,
        )

        start_message = Message(
            role="system",
            name=USER_ROLE_NAME,
            content='hi',
        )
        guest_llm_message = start_message
            
        # Max turns limit and repetition detection
        MAX_TURNS = 20
        turn_count = 0
        recent_guest_messages = []

        while turn_count < MAX_TURNS:
            try:
                turn_count += 1
                
                order_taking_llm_response = await get_order_taking_llm_response(
                    order_taking_llm_handler, guest_llm_message
                )
                order_taker_response_content = order_taking_llm_response.content
                print(
                    f"{ORDER_TAKING_LLM_ROLE_NAME}: {order_taker_response_content}",
                    end="\n",
                    flush=True,
                )

                self.chat_history.append(Message(role="assistant", name=ORDER_TAKING_LLM_ROLE_NAME, content=order_taker_response_content))
                # Conversation with guest LLM
                incoming_message = Message(
                    role="assistant",
                    name=ORDER_TAKING_LLM_ROLE_NAME,
                    content=order_taker_response_content,
                )
                guest_llm_response, guest_llm_metrics = await get_validation_llm_response(
                    guest_llm_handler, incoming_message
                )
                print(
                    f"{VALIDATION_LLM_ROLE_NAME}: {guest_llm_response.content}",
                    end="\n",
                    flush=True,
                )


               
                guest_llm_message = Message(role="agent", name=VALIDATION_LLM_ROLE_NAME, content=guest_llm_response.content, usage=guest_llm_metrics["usage"], latency_ms=guest_llm_metrics["latency_ms"])
                self.chat_history.append(guest_llm_message)
                
                # # Repetition detection - check if guest is asking for confirmation repeatedly
                # recent_guest_messages.append(guest_llm_response.content.lower())
                # if len(recent_guest_messages) > 3:
                #     recent_guest_messages.pop(0)  # Keep only last 3 messages
                    
                #     # Check if last 3 guest messages are asking for similar confirmations
                #     confirmation_keywords = ["confirm", "total", "correct"]
                #     repetitive_confirmations = 0
                #     for msg in recent_guest_messages:
                #         if any(keyword in msg for keyword in confirmation_keywords):
                #             repetitive_confirmations += 1
                    
                #     if repetitive_confirmations >= 3:
                #         print("\n[System: Repetitive confirmation detected, ending conversation]")
                #         self.chat_history.append(Message(role="system", name='system', content="Repetitive confirmation detected, ending conversation"))
                #         break

                # Check if validation LLM is done and wants to end the conversation
                if guest_llm_response.content.lower() in ["quit", "exit", "q"]:
                    break
                
                # Check if max turns reached
                if turn_count >= MAX_TURNS:
                    print(f"\n[System: Maximum conversation length ({MAX_TURNS} turns) reached, ending conversation]")
                    self.chat_history.append(Message(role="system", name='system', content=f"Maximum conversation length ({MAX_TURNS} turns) reached, ending conversation"))
                    break

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                error_message = f"[Error: {e}]"
                self.chat_history.append(Message(role="system", content=error_message))
                print(error_message)
                break
        self.chat_history.append(Message(role="system", name='test_case_info', content=json.dumps(test_case)))
        self.get_chat_messages_log().to_csv(
            f"logs/chat_with_{VALIDATION_LLM_ROLE_NAME}_messages_log_{self.test_case_name}_test_case_number_{self.test_case_index+1}.csv",
            index=False,
        )


if __name__ == "__main__":
    for test_case_index in range(NUM_TEST_CASES):
        for run_iteration_number in range(NUM_RUNS_PER_TEST_CASE):
            conversation_client = ConversationClient()
            conversation_client.test_case_index = test_case_index
            conversation_client.run_iteration_number = run_iteration_number
            asyncio.run(
                conversation_client.run()
            )
