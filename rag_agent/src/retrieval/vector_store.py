import os
from typing import List, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseVectorStoreProvider

class ChromaVectorStoreProvider(BaseVectorStoreProvider):
    """
    Concrete implementation using ChromaDB for storage and Ollama for embeddings.
    """
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="mistral")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def load_from_directory(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Helper method to load PDFs from a directory, split them, and add to the vector store.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            print(f"Created directory {directory_path}. Add PDFs here to load them.")
            return

        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()
        if not docs:
            print(f"No PDFs found in {directory_path}.")
            return
            
        print(f"Loaded {len(docs)} documents from PDF(s).")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        self.add_documents(splits)
        print(f"Added {len(splits)} chunks to ChromaDB.")

    def add_documents(self, documents: List[Document]) -> None:
        if documents:
            self.vector_store.add_documents(documents)

    def get_retriever(self, search_kwargs: dict = None) -> Any:
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
