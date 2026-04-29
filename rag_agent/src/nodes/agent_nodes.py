import re
import io
import sys
import contextlib
import json
import uuid
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from ..core.state import AgentState
from .tools import search_agent_tools, action_agent_tools
from ..core.mcp_manager import MCPManager
import asyncio

class AgentNodes:
    """
    Creates and holds reference to all our sub-agents.
    """
    def __init__(self, llm_provider, vector_store_provider):
        self.llm = llm_provider.get_model()
        self.retriever = vector_store_provider.get_retriever()
        
        # ReAct agents for search and action (tool-calling based)
        self.search_agent = create_react_agent(self.llm, tools=search_agent_tools)
        self.action_agent = create_react_agent(self.llm, tools=action_agent_tools)
        
        # MCP Agent: standardized tool access via Model Context Protocol
        self.mcp_manager = MCPManager()
        self.mcp_agent = None  # Will be initialized lazily or via load_mcp
        self.mcp_tools = []
        
        # Code agent: prompt the LLM to write Python, then exec() it directly
        self.code_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a Python expert. When given a task, respond with ONLY a single fenced python code block "
             "that solves the problem. Use print() to show the result. No explanations, no text outside the code block."
             "IMPORTANT: Ensure your code uses correct indentation (4 spaces per level).\n"
             "Example:\n```python\ndef add(a, b):\n    return a + b\nprint(add(2, 2))\n```"),
            ("human", "{question}")
        ])
        self.code_chain = self.code_prompt | self.llm
        
        # Pure LLM chains for writer and critic
        writer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert creative writer. Fulfill the user's request with flair and creativity."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.writer_agent = writer_prompt | self.llm
        
        critic_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a meticulous critic. Review the previous messages and provide a refined final answer."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.critic_agent = critic_prompt | self.llm

    def _find_tool(self, tool_name: str, tools: List[Any]):
        """Helper to find a tool by name in a list of tools."""
        for t in tools:
            if hasattr(t, "name") and t.name == tool_name:
                return t
        return None

    async def _invoke_agent_async(self, agent, name: str, state: AgentState, tools: List[Any]) -> Dict[str, Any]:
        """Runs an agent asynchronously with manual tool fallback for JSON outputs."""
        print(f"--- ROUTING TO: {name} ---")
        messages_to_return = []
        current_state = state
        
        for attempt in range(2):
            initial_count = len(current_state.get("messages", []))
            try:
                result = await agent.ainvoke(current_state)
            except Exception as e:
                print(f"--- ERROR in {name} ainvoke: {e} ---")
                break

            new_messages = result["messages"][initial_count:]
            if not new_messages:
                break
            
            lastItem = new_messages[-1]
            
            # Check for manual JSON fallback (common in local models)
            if not getattr(lastItem, 'tool_calls', []) and "name" in lastItem.content and "parameters" in lastItem.content:
                print(f"--- WARNING: Model outputted manual JSON tool call in {name} ---")
                json_match = re.search(r'\{.*\}', lastItem.content, re.DOTALL)
                if json_match:
                    try:
                        tool_json = json.loads(json_match.group(0))
                        tool_name = tool_json.get("name")
                        tool_params = tool_json.get("parameters", {})
                        
                        tool = self._find_tool(tool_name, tools)
                        if tool:
                            print(f"--- Manually executing tool: {tool_name} ---")
                            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                            
                            # 1. Create the structured AI message
                            fixed_ai = AIMessage(
                                content=lastItem.content,
                                tool_calls=[{"name": tool_name, "args": tool_params, "id": tool_call_id, "type": "tool_call"}],
                                name=name
                            )
                            
                            # 2. Run the tool
                            tool_result = await tool.ainvoke(tool_params)
                            print(f"--- Tool Result: {str(tool_result)[:200]}... ---")
                            
                            # 3. Create the tool result message
                            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                            
                            # 4. Prepare state for resuming
                            messages_to_return.extend([fixed_ai, tool_msg])
                            current_state = state.copy()
                            current_state["messages"] = state["messages"] + messages_to_return
                            continue # Re-invoke to get final answer
                    except Exception as e:
                        print(f"--- Manual Tool Execution Error: {e} ---")
            
            messages_to_return.extend(new_messages)
            break
        
        if not messages_to_return:
            messages_to_return = [AIMessage(content=f"The {name} was invoked but could not complete the task.", name=name)]
        return {"messages": messages_to_return}

    def _invoke_agent_sync(self, agent, name: str, state: AgentState, tools: List[Any]) -> Dict[str, Any]:
        """Runs an agent synchronously with manual tool fallback."""
        print(f"--- ROUTING TO: {name} ---")
        messages_to_return = []
        current_state = state
        
        for attempt in range(2):
            initial_count = len(current_state.get("messages", []))
            try:
                result = agent.invoke(current_state)
            except Exception as e:
                print(f"--- ERROR in {name} invoke: {e} ---")
                break

            new_messages = result["messages"][initial_count:]
            if not new_messages:
                break
            
            lastItem = new_messages[-1]
            if not getattr(lastItem, 'tool_calls', []) and "name" in lastItem.content and "parameters" in lastItem.content:
                print(f"--- WARNING: Model outputted manual JSON tool call in {name} ---")
                json_match = re.search(r'\{.*\}', lastItem.content, re.DOTALL)
                if json_match:
                    try:
                        tool_json = json.loads(json_match.group(0))
                        tool_name = tool_json.get("name")
                        tool_params = tool_json.get("parameters", {})
                        
                        tool = self._find_tool(tool_name, tools)
                        if tool:
                            print(f"--- Manually executing tool: {tool_name} ---")
                            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                            fixed_ai = AIMessage(
                                content=lastItem.content,
                                tool_calls=[{"name": tool_name, "args": tool_params, "id": tool_call_id, "type": "tool_call"}],
                                name=name
                            )
                            tool_result = tool.invoke(tool_params)
                            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                            
                            messages_to_return.extend([fixed_ai, tool_msg])
                            current_state = state.copy()
                            current_state["messages"] = state["messages"] + messages_to_return
                            continue
                    except Exception as e:
                        print(f"--- Manual Tool Execution Error: {e} ---")
            
            messages_to_return.extend(new_messages)
            break
            
        if not messages_to_return:
            messages_to_return = [AIMessage(content=f"The {name} was invoked but could not complete the task.", name=name)]
        return {"messages": messages_to_return}

    def rag_node(self, state: AgentState) -> Dict[str, Any]:
        """Manually performs RAG using the retriever and the last human message."""
        print(f"--- ROUTING TO: rag_agent ---")
        question = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        
        docs = self.retriever.invoke(question)
        docs_text = "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use ONLY the context below to answer. If the answer is not in the context, say so.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        chain = prompt | self.llm
        res = chain.invoke({"context": docs_text, "question": question})
        return {"messages": [AIMessage(content=res.content, name="rag_agent")], "documents": docs}

    def code_node(self, state: AgentState) -> Dict[str, Any]:
        """Ask the LLM to write Python, then directly exec() it. No tool-calling needed."""
        print(f"--- ROUTING TO: code_agent ---")
        question = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        
        # Step 1: Ask the LLM to write Python code
        response = self.code_chain.invoke({"question": question})
        
        # Step 2: Extract the code block from the markdown response
        code_match = re.search(r"```(?:python)?\s*([\s\S]+?)```", response.content)
        if not code_match:
            # Try to use the raw response as code if no backticks found
            raw = response.content.strip()
            code = raw if raw else None
        else:
            code = code_match.group(1).strip()
        
        if not code:
            return {"messages": [AIMessage(content="I couldn't generate valid Python code for that request.", name="code_agent")]}
        
        # Step 3: Execute the code and capture stdout
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<code_agent>", "exec"), {})
            output = stdout_capture.getvalue().strip()
            result_text = f"The code agent calculated the following result: **{output if output else '(no output)'}**"
        except Exception as e:
            result_text = f"I tried to run a calculation but encountered an error: `{e}`"
        
        return {"messages": [AIMessage(content=result_text, name="code_agent")]}

    def search_node(self, state: AgentState) -> Dict[str, Any]:
        return self._invoke_agent_sync(self.search_agent, "search_agent", state, search_agent_tools)

    def action_node(self, state: AgentState) -> Dict[str, Any]:
        return self._invoke_agent_sync(self.action_agent, "action_agent", state, action_agent_tools)
        
    def writer_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"--- ROUTING TO: writer_agent ---")
        res = self.writer_agent.invoke(state)
        return {"messages": [AIMessage(content=res.content, name="writer_agent")]}
        
    def critic_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"--- ROUTING TO: critic_agent ---")
        res = self.critic_agent.invoke(state)
        return {"messages": [AIMessage(content=res.content, name="critic_agent")]}

    async def mcp_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Handles requests using tools from MCP servers.
        Initializes the agent on first call if not already done.
        """
        if self.mcp_agent is None:
            print("--- Initializing MCP Tools ---")
            try:
                # Run the async tool loading
                self.mcp_tools = await self.mcp_manager.load_all_tools()
                print(f"--- Loaded {len(self.mcp_tools)} tools from MCP servers ---")
                
                if self.mcp_tools:
                    # System prompt with robust instructions for local models
                    mcp_system_prompt = (
                        "You are a HIGH-CAPABILITY tool-access agent with DIRECT INTERNET ACCESS.\n"
                        f"Available tools: {', '.join([t.name for t in self.mcp_tools])}.\n\n"
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. If you need to search or use a tool, respond ONLY with the tool call JSON. \n"
                        "2. DO NOT include any conversational text, thoughts, or apologies. \n"
                        "3. Format tool calls EXACTLY like this: {\"name\": \"web_search\", \"parameters\": {\"query\": \"your query\"}}\n"
                        "4. Once you receive the tool results, then provide your final answer to the user."
                    )
                    self.mcp_agent = create_react_agent(self.llm, tools=self.mcp_tools, prompt=mcp_system_prompt)
                else:
                    return {"messages": [AIMessage(content="No MCP tools were loaded. Please check your MCP server configuration.", name="mcp_agent")]}
            except Exception as e:
                return {"messages": [AIMessage(content=f"Error initializing MCP agent: {e}", name="mcp_agent")]}

        return await self._invoke_agent_async(self.mcp_agent, "mcp_agent", state, self.mcp_tools)


