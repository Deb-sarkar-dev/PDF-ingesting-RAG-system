from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

# 1. Search Tool - Mocked due to local dependency import issues
@tool
def search_tool(query: str) -> str:
    """Searches the internet for information."""
    print(f"--- FAKE SEARCH FOR: {query} ---")
    return f"Search results for {query}: Mocked result (Internet data unavailable)."

# 2. Code Execution Tool
python_repl_tool = PythonAstREPLTool()

# 3. Dummy Action/API Tools
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email. Useful for the API/Action agent."""
    print(f"--- FAKE EMAIL SENT TO {to} ---")
    return f"Success: Email sent to {to} with subject '{subject}'"

@tool
def create_calendar_event(title: str, date: str, time: str) -> str:
    """Creates a calendar event."""
    print(f"--- FAKE CALENDAR EVENT: {title} at {date} {time} ---")
    return f"Success: Created event {title}"

# Export tools based on agent type
search_agent_tools = [search_tool]
code_agent_tools = [python_repl_tool]
action_agent_tools = [send_email, create_calendar_event]
