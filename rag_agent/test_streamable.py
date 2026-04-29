import asyncio
import os
from langchain_mcp_adapters.sessions import StreamableHttpConnection, create_session
from mcp import ClientSession
from dotenv import load_dotenv

async def test_streamable():
    load_dotenv(dotenv_path='rag_agent/.env')
    api_key = os.getenv("EXA_API_KEY")
    url = f"https://mcp.exa.ai/mcp"
    # Note: Streamable HTTP might expect the key in headers or params
    # The guide says it should be in exaApiKey query param
    full_url = f"{url}?exaApiKey={api_key}"
    
    print(f"Connecting to {full_url} using StreamableHttpConnection...")
    
    conn = StreamableHttpConnection(url=full_url)
    try:
        async with create_session(conn) as session:
            print("Session created. Initializing...")
            await session.initialize()
            print("Initialized. Listing tools...")
            tools = await session.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streamable())
