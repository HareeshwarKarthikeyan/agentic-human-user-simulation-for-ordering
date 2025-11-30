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
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from agents.guest_agent import GuestAgentDeps, guest_agent
from agents.message import Message
from agents.order_tracking_agent import OrderTracking
from agents.utils import (
    get_latencies_for_pydantic_agent_interaction,
    run_agent)
from scripts.conversation_config_and_helpers import (
    load_test_case_for_conversation,
    VALIDATION_AGENT_ROLE_NAME,
    USER_ROLE_NAME,
    NUM_TEST_CASES,
    NUM_RUNS_PER_TEST_CASE,
)


class ConversationClient:
    def __init__(self):
        self.chat_history = []
        self.run_iteration_number = 0
        self.test_case_index = 0
        self.test_case_name = ""

    def get_guest_agent_deps_with_test_case(self, test_case_index: int = 0) -> GuestAgentDeps:
        test_case, self.test_case_name = load_test_case_for_conversation(
            test_case_index=test_case_index
        )
        guest_agent_deps = GuestAgentDeps(
            test_case=test_case,
            order_tracking=OrderTracking(
                current_items_in_order=[],
                target_items_to_order=test_case["target_order"]["order_items"]
            ),
            latest_sub_agents_tool_call_latencies=[]
        )
        return guest_agent_deps
    
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
                'usage': json.dumps(message.usage) if message.usage else ''
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
            'usage'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        print(f"Retrieved {len(self.chat_history)} messages as DataFrame")
        return df


    async def run(self):

        guest_agent_deps = self.get_guest_agent_deps_with_test_case(test_case_index=self.test_case_index)


        while True:
            try:
                # Start of conversation with order taking user
                user_input = input("> ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                print(
                    f"Order Taking User: {user_input}",
                    end="\n",
                    flush=True,
                )

                self.chat_history.append(Message(role="user", name=USER_ROLE_NAME, content=user_input))
                guest_agent_response = await run_agent(guest_agent, user_input, guest_agent_deps)
                print(
                    f"{VALIDATION_AGENT_ROLE_NAME}: {guest_agent_response.output}",
                    end="\n",
                    flush=True,
                )

                guest_agent_tool_call_latencies = get_latencies_for_pydantic_agent_interaction(guest_agent_response.new_messages())
                sub_agent_tool_call_latencies = guest_agent_deps.latest_sub_agents_tool_call_latencies
                tool_call_latencies = {
                    "guest_agent": guest_agent_tool_call_latencies,
                    "sub_agents": sub_agent_tool_call_latencies,
                }
                usage = guest_agent_response.usage()
                usage_dict = {
                    "requests": usage.requests,
                    "request_tokens": usage.request_tokens,
                    "response_tokens": usage.response_tokens,
                    "total_tokens": usage.total_tokens,
                }
                self.chat_history.append(Message(role="agent", name=VALIDATION_AGENT_ROLE_NAME, content=guest_agent_response.output, tool_calls=tool_call_latencies, usage=usage_dict))
                guest_agent_deps.latest_sub_agents_tool_call_latencies = []
                

                # Check if validation LLM is done and wants to end the conversation
                if guest_agent_response.output.lower() in ["quit", "exit", "q"]:
                    break

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                error_message = f"[Error: {e}]"
                self.chat_history.append(Message(role="system", content=error_message))
                guest_agent_tool_call_latencies = get_latencies_for_pydantic_agent_interaction(guest_agent_response.new_messages())
                sub_agent_tool_call_latencies = guest_agent_deps.latest_sub_agents_tool_call_latencies
                tool_call_latencies = {
                    "guest_agent": guest_agent_tool_call_latencies,
                    "sub_agents": sub_agent_tool_call_latencies,
                }
                self.chat_history.append(Message(role="agent", name=VALIDATION_AGENT_ROLE_NAME, content="Error", tool_calls=tool_call_latencies))
                print(error_message)
                break

        self.get_chat_messages_log().to_csv(
            f"../validation_conversation_logs/chat_with_{VALIDATION_AGENT_ROLE_NAME}_messages_log_{self.test_case_name}_iteration_{self.run_iteration_number}.csv",
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
