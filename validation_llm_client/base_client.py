from abc import ABC, abstractmethod
from typing import List
from .message import Message


class BaseLLMClient(ABC):
    @abstractmethod
    async def process(self, messages: List[Message]) -> Message:
        """Takes chat history as input and returns assistant reply as Message"""
        pass
