import asyncio
import os
from langchain_ollama import ChatOllama
from src.core.mcp_manager import MCPManager
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

async def test_native_tool_calling():
    load_dotenv(dotenv_path='rag_agent/.env')
    
    # 1. Load tools
    mcp_manager = MCPManager()
    tools = await mcp_manager.load_all_tools()
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    
    # 2. Setup LLM
    llm = ChatOllama(model="llama3.1", temperature=0, base_url="http://127.0.0.1:11434")
    llm_with_tools = llm.bind_tools(tools)
    
    # 3. Test query
    query = "What is the current price of NVIDIA stock?"
    messages = [HumanMessage(content=query)]
    
    print(f"Querying model with native tools...")
    response = await llm_with_tools.ainvoke(messages)
    
    print(f"Response: {response.content}")
    print(f"Tool Calls: {response.tool_calls}")

if __name__ == "__main__":
    asyncio.run(test_native_tool_calling())
