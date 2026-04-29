import operator
from typing import TypedDict, List, Annotated, Any
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    State of the Multi-Agent LangGraph.
    
    Attributes:
        messages: A list of the current conversation messages (using operator.add to append).
        next: The name of the next agent to route to ("__end__" or node name).
        documents: A list of retrieved documents from the vector store (RAG context).
        generation: The final generated answer (for backward compatibility).
    """
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    documents: List[Document]
    generation: str
