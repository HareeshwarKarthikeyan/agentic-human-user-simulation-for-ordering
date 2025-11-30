from typing import List, Dict, Tuple, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
import json
import pandas as pd
from datetime import datetime
from .base_client import BaseLLMClient
from .message import Message


class LLMChatConfig(BaseModel):
    llm_name: str
    system_prompt: str
    associate_prompts: List[str] = Field(default_factory=list)


@dataclass
class LLMChatHandler:
    config: LLMChatConfig
    llm_client: BaseLLMClient
    chat_history: List[Message] = None

    def __init__(self, config: LLMChatConfig, llm_client: BaseLLMClient):
        self.config = config
        self.llm_client = llm_client
        self.chat_history = [Message(role="system", content=self.config.system_prompt)]
        for prompt in self.config.associate_prompts:
            self.chat_history.append(Message(role="system", content=prompt))

    async def chat(self, incoming_message: Message) -> Tuple[Message, Dict[str, Any]]:
        # Add incoming message to chat history
        self.chat_history.append(incoming_message)
        # print('Current chat history: ', self.chat_history)
        # print('Current chat history length: ', len(self.chat_history))
        
        # Get response from LLM client
        assistant_message = await self.llm_client.process(self.chat_history)
        
        # Add assistant response to chat history
        self.chat_history.append(assistant_message)
        
        # Extract metrics from the assistant message
        usage_dict = {
            "num_requests": 1,
            "num_request_tokens": assistant_message.usage.get("prompt_tokens", 0) if assistant_message.usage else 0,
            "num_response_tokens": assistant_message.usage.get("completion_tokens", 0) if assistant_message.usage else 0,
            "total_tokens": assistant_message.usage.get("total_tokens", 0) if assistant_message.usage else 0
        }
        
        metrics = {
            "usage": usage_dict,
            "latency_ms": assistant_message.latency_ms or 0
        }
        
        # Return both the message and metrics
        return assistant_message, metrics

    def reset_conversation(self):
        self.chat_history = [Message(role="system", content=self.config.system_prompt)]
        for prompt in self.config.associate_prompts:
            self.chat_history.append(Message(role="system", content=prompt))

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
                'tool_calls': json.dumps([tool_call for tool_call in message.tool_calls]) if message.tool_calls else '',
                'logs': json.dumps(message.logs) if message.logs else '',
                'trace_id': message.trace_id if message.trace_id else '',
                'parent_id': str(message.parent_id) if message.parent_id else ''
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
            'parent_id'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        print(f"Retrieved {len(self.chat_history)} messages as DataFrame")
        return df

    
