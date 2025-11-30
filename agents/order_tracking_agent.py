from dataclasses import dataclass
import copy
from typing import List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from agents.config import (
    ORDER_TRACKING_AGENT_MODEL, ORDER_TRACKING_AGENT_SYSTEM_INSTRUCTIONS)


class Modifier(BaseModel):
    modifier_type_name: str
    modifier_option_name: str
    is_required: bool
    price: float


class OrderItem(BaseModel):
    name: str
    quantity: int
    base_price: float
    modifiers: List[Modifier] = []
    item_total_price: float


class OrderTracking(BaseModel):
    current_items_in_order: List[OrderItem] = []
    target_items_to_order: List[OrderItem] = []

@dataclass
class OrderTrackingDeps:
   order_tracking: OrderTracking = None


order_tracking_agent = Agent(
    model=ORDER_TRACKING_AGENT_MODEL,
    name='Order Tracking Agent',
    deps_type=OrderTrackingDeps,
    system_prompt = 'You are an agent',
    instructions=ORDER_TRACKING_AGENT_SYSTEM_INSTRUCTIONS,
)

@order_tracking_agent.tool
def look_at_target_items_to_order(ctx: RunContext[OrderTrackingDeps]) -> str:
    return f"Target items to order: {ctx.deps.order_tracking.target_items_to_order}"

@order_tracking_agent.tool
def look_at_items_ordered_so_far(ctx: RunContext[OrderTrackingDeps]) -> str:
    return f"Items ordered so far: {ctx.deps.order_tracking.current_items_in_order}"

@order_tracking_agent.tool
def remove_item_from_order(
    ctx: RunContext[OrderTrackingDeps],
    item_name: str,
    quantity: int,
    modifier_type_names: List[str] = None,
    modifier_option_names: List[str] = None,
    modifier_is_required: List[bool] = None,
    modifier_prices: List[float] = None
):
    """Remove an item from the current order."""
    try:
        # Build modifiers list
        modifiers = []
        if modifier_type_names and modifier_option_names:
            for i in range(len(modifier_type_names)):
                modifiers.append(Modifier(
                    modifier_type_name=modifier_type_names[i],
                    modifier_option_name=modifier_option_names[i],
                    is_required=modifier_is_required[i] if modifier_is_required and i < len(modifier_is_required) else False,
                    price=modifier_prices[i] if modifier_prices and i < len(modifier_prices) else 0.0
                ))
        
        # Find and remove the matching item
        for i, item in enumerate(ctx.deps.order_tracking.current_items_in_order):
            if item.name.lower() == item_name.lower() and item.quantity == quantity:
                ctx.deps.order_tracking.current_items_in_order.pop(i)
                return f"Removed item from order: {item}"
        
        return f"Item '{item_name}' with quantity {quantity} not found in order"
    except Exception as e:
        error_message = f"[Error in remove_item_from_order tool call: {e}]"
        return error_message

@order_tracking_agent.tool
def add_item_to_order(
    ctx: RunContext[OrderTrackingDeps],
    item_name: str,
    quantity: int,
    base_price: float,
    item_total_price: float,
    modifier_type_names: List[str] = None,
    modifier_option_names: List[str] = None,
    modifier_is_required: List[bool] = None,
    modifier_prices: List[float] = None
):
    """Add an item to current order ONLY if it doesn't already exist."""
    # Check if item already exists (avoid duplicates)
    for item in ctx.deps.order_tracking.current_items_in_order:
        if item.name.lower() == item_name.lower():
            # Item already exists, don't add duplicate
            return f"Item '{item_name}' already in order, not adding duplicate"
    
    # Build modifiers list
    modifiers = []
    if modifier_type_names and modifier_option_names:
        for i in range(len(modifier_type_names)):
            modifiers.append(Modifier(
                modifier_type_name=modifier_type_names[i],
                modifier_option_name=modifier_option_names[i],
                is_required=modifier_is_required[i] if modifier_is_required and i < len(modifier_is_required) else False,
                price=modifier_prices[i] if modifier_prices and i < len(modifier_prices) else 0.0
            ))
    
    # Create and add the OrderItem
    new_item = OrderItem(
        name=item_name,
        quantity=quantity,
        base_price=base_price,
        modifiers=modifiers,
        item_total_price=item_total_price
    )
    
    ctx.deps.order_tracking.current_items_in_order.append(new_item)
    return f"Added item to order: {new_item}"

@order_tracking_agent.tool
def clear_current_items_in_order(ctx: RunContext[OrderTrackingDeps]):
    """Clear the current items in order."""
    ctx.deps.order_tracking.current_items_in_order = []
    return f"Cleared current items in order"
