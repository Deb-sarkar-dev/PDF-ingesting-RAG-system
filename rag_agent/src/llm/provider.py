from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    """
    Concrete implementation of the LLM provider using Ollama.
    """
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self._model = ChatOllama(model=self.model_name, temperature=self.temperature, base_url="http://127.0.0.1:11434")

    def get_model(self) -> BaseChatModel:
        """
        Returns an instance of ChatOllama.
        """
        return self._model
