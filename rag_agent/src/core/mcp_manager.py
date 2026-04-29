import asyncio
import os
from contextlib import AsyncExitStack
from typing import List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv

class MCPManager:
    """
    Manages connections to MCP servers and simplifies tool loading for LangChain.
    Keeps connections alive using an AsyncExitStack.
    """
    
    def __init__(self):
        # Load .env from the root of the project (rag_agent directory)
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        load_dotenv(dotenv_path=env_path)
        self.exit_stack = AsyncExitStack()
        self.sessions = []

    async def connect_to_stdio(self, command: str, args: List[str] = None, env: dict = None) -> List[BaseTool]:
        """
        Connects to an MCP server via stdio and returns its tools.
        """
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env={**os.environ, **(env or {})}
        )
        
        try:
            # 1. Connect via stdio
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            
            # 2. Start session
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            
            # 3. Initialize session
            await session.initialize()
            
            # 4. Load tools using the adapter
            tools = await load_mcp_tools(session)
            self.sessions.append(session)
            return tools
        except Exception as e:
            print(f"Error connecting to stdio {command}: {e}")
            return []

    async def connect_to_sse(self, url: str) -> List[BaseTool]:
        """
        Connects to an MCP server via SSE and returns its tools.
        """
        try:
            # 1. Connect via SSE
            read, write = await self.exit_stack.enter_async_context(sse_client(url))
            
            # 2. Start session
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            
            # 3. Initialize session
            await session.initialize()
            
            # 4. Load tools using the adapter
            tools = await load_mcp_tools(session)
            self.sessions.append(session)
            return tools
        except Exception as e:
            print(f"Error connecting to SSE {url}: {e}")
            return []

    async def load_all_tools(self) -> List[BaseTool]:
        """
        Loads tools from all configured MCP servers.
        """
        all_tools = []
        
        # 1. Load Local Exa Search MCP (stdio)
        exa_server_path = os.path.join(os.path.dirname(__file__), "mcp_server_exa.py")
        if os.path.exists(exa_server_path):
            import sys
            python_exe = sys.executable
            abs_exa_path = os.path.abspath(exa_server_path)
            exa_tools = await self.connect_to_stdio(python_exe, [abs_exa_path])
            all_tools.extend(exa_tools)
        else:
            print("Warning: mcp_server_exa.py not found. Skipping Exa MCP.")

        # 2. Load Local Sample MCP (stdio)
        sample_server_path = os.path.join(os.path.dirname(__file__), "mcp_server_sample.py")
        if os.path.exists(sample_server_path):
            import sys
            python_exe = sys.executable
            # Ensure we use absolute path for the script
            abs_sample_path = os.path.abspath(sample_server_path)
            local_tools = await self.connect_to_stdio(python_exe, [abs_sample_path])
            all_tools.extend(local_tools)
            
        return all_tools

    async def close(self):
        """Closes all active connections."""
        await self.exit_stack.aclose()
