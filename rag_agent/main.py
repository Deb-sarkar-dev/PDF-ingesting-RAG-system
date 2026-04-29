import os
import argparse
from src.llm.provider import OllamaProvider
from src.retrieval.vector_store import ChromaVectorStoreProvider
from src.agent.graph import LangGraphAgent

import asyncio

async def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Agent with Ollama and Chroma")
    parser.add_argument("--query", type=str, help="Question to ask the agent", default="What are the main topics in the documents?")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF files to ingest", default="rag_agent/data")
    args = parser.parse_args()

    print("\n--- Initializing Providers ---")
    # Using OOP interfaces
    llm_provider = OllamaProvider(model_name="llama3.2")
    vector_store_provider = ChromaVectorStoreProvider(persist_directory="./chroma_db", collection_name="rag_agent_collection")

    print(f"\n--- Checking for PDFs in '{args.pdf_dir}' ---")
    vector_store_provider.load_from_directory(args.pdf_dir)

    print("\n--- Building LangGraph Agent ---")
    agent = LangGraphAgent(llm_provider, vector_store_provider)
    app = agent.graph

    print("\n--- Running Inference ---")
    print(f"Question: {args.query}\n")
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content=args.query)]}
    
    # We can stream intermediate outputs from node state modifications
    async for output in app.astream(inputs):
        for key, value in output.items():
            print(f"[{key}] node completed.")
            if "messages" in value and value["messages"]:
                msg = value["messages"][-1]
                print(f"Agent {msg.name or key} says: {msg.content}")
            
    # Then access the final accumulated answers
    final_state = await app.ainvoke(inputs)
    print("\n=== FINAL ANSWER ===")
    messages = final_state.get("messages", [])
    if messages:
        print(messages[-1].content)
    else:
        print("No generation produced.")

if __name__ == "__main__":
    asyncio.run(main())