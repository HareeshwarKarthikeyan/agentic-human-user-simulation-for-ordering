import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path so we can import from other parent directories
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from validation_llm_client.message import Message
from scripts.conversation_config_and_helpers import (
    setup_validation_llm_handler,
    get_validation_llm_response,
    load_test_case_for_conversation,
    VALIDATION_LLM_ROLE_NAME,
    USER_ROLE_NAME,
    NUM_TEST_CASES,
    NUM_RUNS_PER_TEST_CASE,
)



async def run(run_iteration_number: int = 0, test_case_index: int = 0):

    # adjust the parameters for all the function calls here according to the edits made to the function definitions in the conversation_config_and_helpers.py file

    test_case, test_case_name = load_test_case_for_conversation(
        test_case_index=test_case_index
    )

    validation_llm_handler = setup_validation_llm_handler(
        provider="openai",
        llm_role_name=VALIDATION_LLM_ROLE_NAME + "-" + test_case_name,
        test_case=test_case,
    )

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

            # Conversation with validation LLM
            incoming_message = Message(
                role="user",
                name=USER_ROLE_NAME,
                content=user_input,
            )
            validation_llm_response, _ = await get_validation_llm_response(
                validation_llm_handler, incoming_message
            )
            print(
                f"{VALIDATION_LLM_ROLE_NAME}: {validation_llm_response.content}",
                end="\n",
                flush=True,
            )

            # Check if validation LLM is done and wants to end the conversation
            if validation_llm_response.content.lower() in ["quit", "exit", "q"]:
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            error_message = f"[Error: {e}]"
            validation_llm_handler.chat_history.append(Message(role="system", content=error_message))
            print(error_message)
            break

    validation_llm_handler.get_chat_messages_log().to_csv(
        f"../validation_conversation_logs/chat_with_{VALIDATION_LLM_ROLE_NAME}_messages_log_{test_case_name}_iteration_{run_iteration_number}.csv",
        index=False,
    )


if __name__ == "__main__":
    for test_case_index in range(NUM_TEST_CASES):
        for run_iteration_number in range(NUM_RUNS_PER_TEST_CASE):
            asyncio.run(
                run(
                    run_iteration_number=run_iteration_number,
                    test_case_index=test_case_index,
                )
            )
