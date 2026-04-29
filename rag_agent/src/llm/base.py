from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel

class BaseLLMProvider(ABC):
    """
    Abstract base class defining the interface for an LLM provider.
    Ensures easy swapping between different LLM implementations.
    """

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """
        Returns a LangChain BaseChatModel instance.
        """
        pass
