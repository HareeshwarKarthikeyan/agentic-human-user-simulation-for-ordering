import sys
from pathlib import Path
import asyncio
import json

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path so we can import from other parent directories
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from validation_llm_client.message import Message
from scripts.conversation_config_and_helpers import (
    setup_order_taking_llm_handler,
    get_order_taking_llm_response,
    ORDER_TAKING_LLM_ROLE_NAME,
    USER_ROLE_NAME,
   
)


def load_menu_string(menu_path: str):
    """Load menu from menu.json file and return as JSON string"""
    with open(menu_path, 'r') as f:
        menu_data = json.load(f)
        return json.dumps(menu_data, indent=2)



async def run():

    menu_string = load_menu_string("../test_case_generation/data/menu.json")
    order_taking_llm_handler = setup_order_taking_llm_handler(
        provider="openai",
        llm_role_name=ORDER_TAKING_LLM_ROLE_NAME,
        menu_string=menu_string,
    )


    while True:
        try:
            # Start of conversation with order taking user
            user_input = input("> ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            print(
                f"{USER_ROLE_NAME}: {user_input}",
                end="\n",
                flush=True,
            )

            # Conversation with validation LLM
            incoming_message = Message(
                role="user",
                name=USER_ROLE_NAME,
                content=user_input,
            )
            order_taking_llm_response = await get_order_taking_llm_response(
                order_taking_llm_handler, incoming_message
            )
            print(
                f"{ORDER_TAKING_LLM_ROLE_NAME}: {order_taking_llm_response.content}",
                end="\n",
                flush=True,
            )

            # Check if validation LLM is done and wants to end the conversation
            if order_taking_llm_response.content.lower() in ["quit", "exit", "q"]:
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            error_message = f"[Error: {e}]"
            order_taking_llm_handler.chat_history.append(Message(role="system", content=error_message))
            print(error_message)
            break



if __name__ == "__main__":
    asyncio.run(run())
