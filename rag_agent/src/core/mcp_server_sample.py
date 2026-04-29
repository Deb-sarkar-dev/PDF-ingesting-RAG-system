from mcp.server.fastmcp import FastMCP
import datetime
import platform
import os

# Create an MCP server
mcp = FastMCP("SampleServer")

@mcp.tool()
def get_system_time() -> str:
    """Returns the current system time."""
    return f"Current system time is: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def get_os_info() -> str:
    """Returns information about the operating system."""
    return f"OS: {platform.system()} {platform.release()}, Machine: {platform.machine()}"

@mcp.tool()
def get_working_directory() -> str:
    """Returns the current working directory."""
    return f"Current working directory: {os.getcwd()}"

if __name__ == "__main__":
    mcp.run()
