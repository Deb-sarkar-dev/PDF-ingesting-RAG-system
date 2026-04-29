from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document

class BaseVectorStoreProvider(ABC):
    """
    Abstract base class defining the interface for vector store interactions.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to the vector store."""
        pass

    @abstractmethod
    def get_retriever(self, search_kwargs: dict = None) -> Any:
        """Returns the retriever instance."""
        pass
