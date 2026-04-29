import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.llm.provider import OllamaProvider
from src.retrieval.vector_store import ChromaVectorStoreProvider
from src.agent.graph import LangGraphAgent

from dotenv import load_dotenv

# Load for Exa/MCP
load_dotenv()

# Global agent reference
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup LangGraph RAG Agent on startup
    global agent_app
    print("Initializing Providers for API...")
    llm_provider = OllamaProvider(model_name="llama3.2")
    vector_store_provider = ChromaVectorStoreProvider(persist_directory="./chroma_db", collection_name="rag_agent_collection")
    
    print("Checking for documents in rag_agent/data...")
    vector_store_provider.load_from_directory("rag_agent/data")
    
    rag_agent = LangGraphAgent(llm_provider, vector_store_provider)
    agent_app = rag_agent.graph
    print("API is ready to serve predictions.")
    yield
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    from langchain_core.messages import HumanMessage
    if not agent_app:
        return {"error": "Agent not initialized"}
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    try:
        # Must use ainvoke for MCP/Async Graph support
        final_state = await agent_app.ainvoke(inputs)
        messages = final_state.get("messages", [])
        if messages:
            # Last message is the AI response
            final_content = messages[-1].content
        else:
            final_content = "I'm sorry, I couldn't generate an answer."
        return {"response": final_content}
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        return {"error": str(e)}

# Serve static files for frontend dynamically
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(frontend_path, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    # Start the local server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
