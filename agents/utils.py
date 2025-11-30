from typing import Any, Dict, List

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart



async def run_agent(agent: Agent, input: str,deps,message_history: list = []):
    try:
        result = await agent.run(
            input,
            deps=deps,
            message_history= message_history,
        )
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        print('cause:', repr(e.__cause__))
    return result


def get_latencies_for_pydantic_agent_interaction(messages: List[ModelMessage]) -> Dict[str, Any]:
    result = {
        "tool_calls": [],
        "total_latency_sec": None
    }

    user_prompt_time = None
    tool_calls = {}  
    last_response_time = None

    for msg in messages:
        if msg.kind == "request":
            for part in msg.parts:
                if part.part_kind == "user-prompt" and not user_prompt_time:
                    user_prompt_time = part.timestamp
                    last_response_time = user_prompt_time
                elif part.part_kind == "tool-return":
                    tool_call_id = getattr(part, "tool_call_id", None)
                    if tool_call_id:
                        if tool_call_id in tool_calls:
                            tool_calls[tool_call_id]["return_time"] = part.timestamp
                            tool_calls[tool_call_id]["result"] = part.content
                        else:
                            tool_calls[tool_call_id] = {
                                "tool_name": None,
                                "call_time": None,
                                "return_time": part.timestamp,
                                "result": part.content,
                            }
                        last_response_time = max(last_response_time, part.timestamp)
        elif msg.kind == "response":
            for part in msg.parts:
                if part.part_kind == "tool-call":
                    tool_call_id = getattr(part, "tool_call_id", None)
                    tool_name = getattr(part, "tool_name", "unknown_tool")
                    if tool_call_id:
                        tool_calls[tool_call_id] = {
                            "tool_name": tool_name,
                            "call_time": msg.timestamp,
                            "return_time": None,
                            "args": part.args,
                        }
                elif part.part_kind == "text":
                    last_response_time = max(last_response_time, msg.timestamp)
    
    # Compute latencies
    for call_id, data in tool_calls.items():
        call_time = data["call_time"]
        return_time = data.get("return_time")
        call_time = max(call_time, user_prompt_time)
        decision_latency = (call_time - user_prompt_time).total_seconds() if user_prompt_time else None 
        execution_latency = (return_time - call_time).total_seconds() if return_time else None
        tool_calls[call_id]["decision_latency_sec"] = decision_latency
        tool_calls[call_id]["execution_latency_sec"] = execution_latency

    result["tool_calls"] = tool_calls
    result["last_response_time"] = last_response_time
    result["user_prompt_time"] = user_prompt_time

    if last_response_time and user_prompt_time:
        result["total_latency_sec"] = (last_response_time - user_prompt_time).total_seconds()
    
    return result


def get_last_user_prompt_part(messages: List[ModelMessage]) -> str:
    model_request_messages =  [msg for msg in messages if isinstance(msg, ModelRequest)]
    model_request_message_user_prompt_parts = [part for msg in model_request_messages for part in msg.parts if isinstance(part, UserPromptPart)]
    last_model_message_user_prompt_part = model_request_message_user_prompt_parts[-1]
    return last_model_message_user_prompt_part
