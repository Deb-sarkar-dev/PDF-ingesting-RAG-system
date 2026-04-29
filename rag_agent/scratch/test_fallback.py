import asyncio
import json
import uuid
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage

# Mock AgentState
class AgentState(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "messages" not in self:
            self["messages"] = []

# Mock Agent (inner react graph)
class MockAgent:
    def __init__(self):
        self.call_count = 0
    
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        self.call_count += 1
        # Behavior like a real LangGraph: result contains state after execution
        messages = list(state.get("messages", []))
        if self.call_count == 1:
            # First call: Return manual JSON text
            messages.append(AIMessage(content='{"name": "web_search", "parameters": {"query": "nvidia stock price"}}'))
        else:
            # Second call: Return success
            messages.append(AIMessage(content="NVIDIA stock is at $188."))
        return {"messages": messages}

# The function to test (copied logic)
async def agent_node_async(state: AgentState, agent, name: str) -> Dict[str, Any]:
    print(f"--- ROUTING TO: {name} ---")
    
    # Track which messages we've already added to avoid duplicates if re-invoking
    current_state = state.copy()
    
    for attempt in range(2):
        print(f"--- Attempt {attempt + 1} ---")
        initial_messages_count = len(current_state.get("messages", []))
        result = await agent.ainvoke(current_state)
        
        if "messages" in result and len(result["messages"]) > initial_messages_count:
            lastItem = result["messages"][-1]
            print(f"Last item content: {lastItem.content[:50]}...")
            
            if not getattr(lastItem, 'tool_calls', []) and '"name"' in lastItem.content and '"parameters"' in lastItem.content:
                print(f"--- Detected manual JSON ---")
                import re
                json_match = re.search(r'\{.*\}', lastItem.content, re.DOTALL)
                if json_match:
                    try:
                        tool_json = json.loads(json_match.group(0))
                        if "name" in tool_json and "parameters" in tool_json:
                            print(f"--- Target tool: {tool_json['name']} ---")
                            tool_call = {
                                "name": tool_json["name"],
                                "args": tool_json["parameters"],
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "type": "tool_call"
                            }
                            new_msg = AIMessage(content=lastItem.content, tool_calls=[tool_call], name=name)
                            # Update current_state for the next call
                            current_state["messages"] = list(result["messages"][:-1]) + [new_msg]
                            continue 
                    except Exception as e:
                        print(f"Error parsing JSON: {e}")
            
            # If we reach here, we are done
            return {"messages": [lastItem]}
        break
    return {"messages": [AIMessage(content="Fail", name=name)]}

async def test():
    state = AgentState(messages=[HumanMessage(content="What is NVIDIA stock price?")])
    agent = MockAgent()
    
    print("Testing fallback mechanism...")
    result = await agent_node_async(state, agent, "mcp_agent")
    
    print(f"\nFinal Result: {result['messages'][-1].content}")
    print(f"Agent Call Count: {agent.call_count}")
    
    if agent.call_count == 2:
        print("\nSUCCESS: The agent was re-invoked after JSON detection!")
    else:
        print("\nFAILURE: Fallback logic did not trigger re-invocation.")

if __name__ == "__main__":
    asyncio.run(test())
