from langgraph.prebuilt import create_react_agent
# Try to instantiate with state_modifier to see if it errors
try:
    # We don't need a real LLM for this check, just seeing if the constructor accepts the keyword
    create_react_agent(None, tools=[], state_modifier="test")
    print("state_modifier is accepted")
except TypeError as e:
    print(f"state_modifier error: {e}")

try:
    create_react_agent(None, tools=[], prompt="test")
    print("prompt is accepted")
except TypeError as e:
    print(f"prompt error: {e}")
