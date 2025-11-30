from typing import List, Dict
import time
from openai import OpenAI
from .base_client import BaseLLMClient
from .message import Message

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str, llm_role_name: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.llm_role_name = llm_role_name

    async def process(self, messages: List[Message]) -> Message:
        # Convert Message objects to OpenAI API format
        chat_messages = []
        for msg in messages:
            # Map custom roles to OpenAI-supported roles
            openai_role = msg.role
            if msg.role == "agent":
                openai_role = "assistant"  # Map agent role to user for guest simulation
            
            chat_message = {
                "role": openai_role,
                "content": msg.content
            }
            # Add name if present and role is appropriate
            if msg.name and msg.role in ["user", "assistant", "agent"]:
                chat_message["name"] = msg.name
            chat_messages.append(chat_message)
        
        # Record start time for latency measurement
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages
        )
        
        # Calculate latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract usage information
        usage_data = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        # Create and return a Message object
        return Message(
            role="user",
            content=response.choices[0].message.content,
            name=self.llm_role_name,
            usage=usage_data,
            latency_ms=latency_ms
        )
