from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.state import AgentState

class RAGNodes:
    """
    Contains the LangGraph node logic formatted as methods.
    Receives dependencies via the constructor.
    """
    def __init__(self, llm_provider, vector_store_provider):
        self.llm = llm_provider.get_model()
        self.retriever = vector_store_provider.get_retriever()
        
    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents from the vector store based on the question.
        """
        question = state["question"]
        print(f"---RETRIEVE for question: '{question}'---")
        documents = self.retriever.invoke(question)
        return {"documents": documents}
        
    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate answer using RAG.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Format docs into a single string context
        docs_text = "\n\n".join(doc.page_content for doc in documents)
        
        # RAG Prompt template
        prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know.\n\n"
            "Question: {question}\n\n"
            "Context: {context}\n\n"
            "Answer:"
        )
        
        # Create pipeline: prompt -> llm -> get string response
        rag_chain = prompt | self.llm | StrOutputParser()
        
        generation = rag_chain.invoke({"context": docs_text, "question": question})
        return {"generation": generation}
