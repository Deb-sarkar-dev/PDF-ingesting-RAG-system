import asyncio
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

async def debug_exa():
    load_dotenv(dotenv_path='rag_agent/.env')
    api_key = os.getenv("EXA_API_KEY")
    url = f"https://mcp.exa.ai/mcp?exaApiKey={api_key}"
    print(f"Connecting to {url}...")
    
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
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
    asyncio.run(debug_exa())
