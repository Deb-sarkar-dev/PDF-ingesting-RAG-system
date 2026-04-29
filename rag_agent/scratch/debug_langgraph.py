import asyncio
import sys
import traceback
from rag_agent.src.llm.provider import OllamaProvider
from rag_agent.src.retrieval.vector_store import ChromaVectorStoreProvider
from rag_agent.src.agent.graph import LangGraphAgent
from langchain_core.messages import HumanMessage

async def main():
    try:
        llm_provider = OllamaProvider(model_name="llama3.2")
        vector_store_provider = ChromaVectorStoreProvider(persist_directory="./chroma_db", collection_name="rag_agent_collection")
        agent = LangGraphAgent(llm_provider, vector_store_provider)
        app = agent.graph
        
        query = "current NVIDIA stock price"
        inputs = {"messages": [HumanMessage(content=query)]}
        
        print("Running agent...")
        async for output in app.astream(inputs):
            print(f"Step: {output}")
            
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
