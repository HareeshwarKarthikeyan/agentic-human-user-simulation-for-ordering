# All the imports for the setup and access of the candidate llm agent to follow the 'Message' structure of the validation llm agent for agent to agent conversation happen here

import sys
import os
from pathlib import Path
import json

# Add the parent directory to the Python path so we can import from other parent directories
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


from dotenv import load_dotenv
from scripts.llm_prompts_config import (
    guest_llm_system_prompt,
    guest_llm_associate_system_prompt,
    order_taking_llm_system_prompt
)
from test_case_generation.test_case_loader import TestCaseLoader

from validation_llm_client.chat_handler import (
    LLMChatConfig,
    LLMChatHandler,
)
from validation_llm_client.openai_client import OpenAIClient
from validation_llm_client.message import Message


# this path must point to a json list of test cases (which follow the class structures in the test_case_generation module)
TEST_CASE_FILE_PATH = "../../test_case_generation/data/order_test_cases.json"
# this constant points to the restaurant guid to test (follows the class structures in the test_case_generation module)
TEST_GUID = "3997923f-e407-4a7f-aa97-cd5c19cbefee"


# edit these constants to define the test case info for the conversation
VALIDATION_AGENT_ROLE_NAME = "guest_simulation_agent"
VALIDATION_LLM_ROLE_NAME = "guest_simulation_llm"
USER_ROLE_NAME = "order_taking_user"

NUM_TEST_CASES = 60  # edit this to get the number of test cases you want to test from the test case file
NUM_RUNS_PER_TEST_CASE = 1  # edit this to get the number of runs you want to do for each test case

ORDER_TAKING_LLM_ROLE_NAME = "order_taking_llm"


# need to be adjusted based on the custom defined test case loader class to load the desired test case from the test case file
def load_test_case_for_conversation(test_case_index: int = 0) -> tuple[dict, str]:
    test_case = json.loads(
        TestCaseLoader(file_path=TEST_CASE_FILE_PATH).get_test_case_by_index(
            test_case_index, remove_guids=True
        )
    )
    test_case_name = test_case["guest_persona"]["full_name"].replace(" ", "-")
    return test_case, test_case_name



# need to be adjusted based on the custom defined test cases and prompts for the validation llm and the desired llm provider client
# the handler returned by this function is only used to get the response from the validation llm in the get_validation_llm_response function
def setup_validation_llm_handler(
    provider: str, llm_role_name: str, test_case: dict
) -> LLMChatHandler:
    if provider == "openai":
        llm_client = OpenAIClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            llm_role_name=llm_role_name,
        )
    else:
        raise ValueError("Unsupported provider")

    config = LLMChatConfig(
        llm_name=llm_role_name,
        system_prompt=guest_llm_system_prompt.format(
            guest_persona=test_case["guest_persona"],
            target_order=test_case["target_order"],
        ),
        associate_prompts=[guest_llm_associate_system_prompt],
    )

    handler = LLMChatHandler(config=config, llm_client=llm_client)
    return handler


# this function doesnt require modifications for most of the cases as the validation llm handler is already configured with the test case and prompts
# call the internal processing and tool call logs from the validation llm handler to get the response and tool calls
# this function is used to get the response from the validation llm for the agent to agent conversation between the candidate and validation llm
async def get_validation_llm_response(
    validation_llm_handler: LLMChatHandler, incoming_message: Message
) -> Message:
    validation_llm_response, validation_llm_metrics = await validation_llm_handler.chat(incoming_message)
    return validation_llm_response, validation_llm_metrics

def setup_order_taking_llm_handler(
    provider: str, llm_role_name: str, 
    menu_string: str,
) -> LLMChatHandler:
    if provider == "openai":
        llm_client = OpenAIClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            llm_role_name=llm_role_name,
        )
    else:
        raise ValueError("Unsupported provider")

    config = LLMChatConfig(
        llm_name=llm_role_name,
        system_prompt=order_taking_llm_system_prompt.format(menu=menu_string),
    )

    handler = LLMChatHandler(config=config, llm_client=llm_client)
    return handler


async def get_order_taking_llm_response(
    order_taking_llm_handler: LLMChatHandler, incoming_message: Message
) -> Message:
    order_taking_llm_response, _ = await order_taking_llm_handler.chat(incoming_message)
    return order_taking_llm_response



    