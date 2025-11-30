import json
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.config import (GUEST_AGENT_MODEL,
                            GUEST_AGENT_SYSTEM_INSTRUCTIONS)
from agents.ablation_config import ( 
                                    GUEST_AGENT_WITHOUT_ORDER_TRACKING_AND_MESSAGE_ATTRIBUTES_GENERATION_AGENT_INSTRUCTIONS,
                                    GUEST_AGENT_WITH_ONLY_ORDER_TRACKING_AGENT_INSTRUCTIONS,
                                    GUEST_AGENT_WITH_ONLY_MESSAGE_ATTRIBUTES_GENERATION_AGENT_INSTRUCTIONS
                                    )
from agents.message_generation_agent import (
    IsOrderingComplete, MessageAttributesGenerationDeps,
    MessageGenerationAttributes, message_attributes_generation_agent)
from agents.order_tracking_agent import (OrderTracking,
                                                          OrderTrackingDeps,
                                                          order_tracking_agent,
                                                          OrderItem,
                                                          Modifier)
from agents.utils import (
    get_last_user_prompt_part, get_latencies_for_pydantic_agent_interaction,
    run_agent)


@dataclass
class GuestAgentDeps:
    test_case: dict
    order_tracking: OrderTracking = None
    next_message_generation_attributes: MessageGenerationAttributes = None
    latest_sub_agents_tool_call_latencies: List[Dict[str, Any]] = Field(default_factory=list)
    past_messages_history:str = ""

guest_agent = Agent(
    model=GUEST_AGENT_MODEL,
    name='Guest Agent',
    deps_type=GuestAgentDeps,
    output_type=str,
    system_prompt = 'You are an agent',
    # instructions=GUEST_AGENT_WITHOUT_ORDER_TRACKING_AND_MESSAGE_ATTRIBUTES_GENERATION_AGENT_INSTRUCTIONS, # use this for exp2 ablation experiment
    # instructions=GUEST_AGENT_WITH_ONLY_ORDER_TRACKING_AGENT_INSTRUCTIONS, # use this for exp3 ablation experiment
    # instructions=GUEST_AGENT_WITH_ONLY_MESSAGE_ATTRIBUTES_GENERATION_AGENT_INSTRUCTIONS, # use this for exp4 ablation experiment
    instructions=GUEST_AGENT_SYSTEM_INSTRUCTIONS, # use this for exp5 ablation experiment

)

# always keep this tool for all ablation experiments
@guest_agent.tool
def get_name_of_self(ctx: RunContext[GuestAgentDeps]) -> str:
    return ctx.deps.test_case["guest_persona"]["full_name"]

# comment out this tool for exp2 and exp4 ablation experiments - there is no order tracking agent in these experiments
@guest_agent.tool
async def update_order_tracking(ctx: RunContext[GuestAgentDeps]) -> str:
    ordering_agent_deps = OrderTrackingDeps(
        order_tracking=ctx.deps.order_tracking
    )
    latest_response_from_restaurant = get_last_user_prompt_part(ctx.messages).content
    input = f"Update the order tracking based on the following latest message from the restaurant: {latest_response_from_restaurant}"
    result = await run_agent(order_tracking_agent, input, ordering_agent_deps)
    ctx.deps.latest_sub_agents_tool_call_latencies.append(get_latencies_for_pydantic_agent_interaction(result.new_messages()))
    # print(f"\nctx.deps.order_tracking: {ctx.deps.order_tracking}\n")
    # print(f"\nlatest tool call latencies: {ctx.deps.latest_sub_agents_tool_call_latencies}\n")
    return f"Latest Order tracking : {ctx.deps.order_tracking}"

# # comment out this tool for exp4 and exp5 ablation experiment where the persona comes from the message_attributes_generation_agent 
# # keep this tool for exp2 and exp3 ablation experiments - there is no message attributes generation agent in these experiments
# @guest_agent.tool
# def get_persona_of_self(ctx: RunContext[GuestAgentDeps]) -> str:
#     return ctx.deps.test_case["guest_persona"]["guest_bio"]

# comment out this tool for exp2 and exp3 ablation experiments - there is no message attributes generation agent in these experiments
@guest_agent.tool
async def get_message_generation_attributes_for_next_message(ctx: RunContext[GuestAgentDeps]) -> str:
    message_attributes_generation_agent_deps = MessageAttributesGenerationDeps(
        guest_personality_description=ctx.deps.test_case["guest_persona"]["guest_bio"],
        order_tracking=ctx.deps.order_tracking
    )
    latest_response_from_restaurant = get_last_user_prompt_part(ctx.messages).content
    input = f"Determine the message attributes for the next message to be generated based on the following latest response from the restaurant: {latest_response_from_restaurant}"
    result = await run_agent(message_attributes_generation_agent, input, message_attributes_generation_agent_deps)
    ctx.deps.next_message_generation_attributes = result.output
    ctx.deps.latest_sub_agents_tool_call_latencies.append(get_latencies_for_pydantic_agent_interaction(result.new_messages()))
    # print(f"\nctx.deps.next_message_generation_attributes: {ctx.deps.next_message_generation_attributes}\n")
    # print(f"\nlatest tool call latencies: {ctx.deps.latest_sub_agents_tool_call_latencies}\n")
    return f"Message generation attributes for your next message: {ctx.deps.next_message_generation_attributes}"


# # comment out this tool for exp3 and exp5 ablation experiments
# # this is only used for exp2 when there are no sub agents and exp4 when there is no order tracking for the message attributes generation agent to id order completion
# @guest_agent.tool
# def get_target_order(ctx: RunContext[GuestAgentDeps]) -> list[dict]:
#     return ctx.deps.test_case["target_order"]

# # comment out this tool for exp3 and exp5 ablation experiments
# # this is only used for exp2 when there are no sub agents and exp4 when there is no order tracking for the message attributes generation agent to id order completion
# @guest_agent.tool
# def get_past_messages_history(ctx: RunContext[GuestAgentDeps]) -> str:
#     return ctx.deps.past_messages_history






