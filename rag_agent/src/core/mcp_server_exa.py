from mcp.server.fastmcp import FastMCP
from exa_py import Exa
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Exa client
exa_api_key = os.getenv("EXA_API_KEY")
if not exa_api_key:
    raise ValueError("EXA_API_KEY not found in .env file")

exa = Exa(api_key=exa_api_key)

# Create MCP server
mcp = FastMCP("ExaSearch")

@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> str:
    """
    Performs a neural web search using Exa.
    Use this for real-time data, news, and finding relevant URLs.
    """
    try:
        results = exa.search(
            query, 
            num_results=num_results,
            type="auto"
        )
        
        output = []
        for res in results.results:
            output.append(f"Title: {res.title}\nURL: {res.url}\nID: {res.id}\n---")
        
        return "\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error during Exa search: {e}"

@mcp.tool()
async def get_page_content(url: str) -> str:
    """
    Retrieves the full text content from a specific URL.
    Use this when you have a specific URL (from search) and need to read its content.
    """
    try:
        results = exa.get_contents([url], text={"max_characters": 10000})
        if results.results:
            res = results.results[0]
            return f"Title: {res.title}\nContent:\n{res.text}"
        return "No content could be extracted from this URL."
    except Exception as e:
        return f"Error retrieving page content: {e}"

if __name__ == "__main__":
    mcp.run()
