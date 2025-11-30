from dataclasses import dataclass
from enum import Enum
import copy

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from agents.config import (
    MESSAGE_ATTRIBUTES_GENERATION_AGENT_MODEL,
    MESSAGE_ATTRIBUTES_GENERATION_AGENT_SYSTEM_INSTRUCTIONS)
from agents.order_tracking_agent import OrderTracking


class MessageMoodTone(Enum):
    CASUAL = 'casual'
    FRUSTRATED = 'frustrated'
    CONFUSED = 'confused'
    ENTHUSIASTIC = 'enthusiastic'

class OrderingStyle(Enum):
    ONE_BY_ONE = 'one_by_one'
    ALL_AT_ONCE = 'all_at_once'

class MenuExplorationStyle(Enum):
    DOES_NOT_EXPLORE_MENU = 'does_not_explore_menu'
    EXPLORES_MENU = 'explores_menu'

class IsOrderingComplete(Enum):
    YES = 'yes'
    NO = 'no'

@dataclass
class MessageAttributesGenerationDeps:
    guest_personality_description: str
    order_tracking: OrderTracking

class MessageGenerationAttributes(BaseModel):
    is_ordering_complete: IsOrderingComplete = IsOrderingComplete.NO
    next_message_menu_exploration_style: MenuExplorationStyle = MenuExplorationStyle.DOES_NOT_EXPLORE_MENU
    next_message_mood_tone: MessageMoodTone = MessageMoodTone.CASUAL
    next_message_ordering_style: OrderingStyle = OrderingStyle.ONE_BY_ONE
    guest_personality_description: str

message_attributes_generation_agent = Agent(
    model=MESSAGE_ATTRIBUTES_GENERATION_AGENT_MODEL,
    name='Message Attributes Generation Agent',
    deps_type=MessageAttributesGenerationDeps,
    output_type=MessageGenerationAttributes,
    system_prompt = 'You are an agent',
    instructions=MESSAGE_ATTRIBUTES_GENERATION_AGENT_SYSTEM_INSTRUCTIONS,
)

@message_attributes_generation_agent.tool
def get_guest_personality_description(ctx: RunContext[MessageAttributesGenerationDeps]) -> str:
    return f"Guest personality description: {ctx.deps.guest_personality_description}"

@message_attributes_generation_agent.tool
def get_items_ordered_so_far(ctx: RunContext[MessageAttributesGenerationDeps]) -> str:
    return f"Items ordered so far: {ctx.deps.order_tracking.current_items_in_order}"

@message_attributes_generation_agent.tool
def get_target_items_to_order(ctx: RunContext[MessageAttributesGenerationDeps]) -> str:
    return f"Target items to order: {ctx.deps.order_tracking.target_items_to_order}"