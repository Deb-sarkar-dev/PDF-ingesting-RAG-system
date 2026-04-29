from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..core.state import AgentState
from ..nodes.agent_nodes import AgentNodes

class RouteDecision(BaseModel):
    """The decision of where to route next."""
    next: Literal["rag_agent", "search_agent", "code_agent", "action_agent", "writer_agent", "critic_agent", "mcp_agent", "FINISH"] = Field(
        ..., description="The next agent to route to, or FINISH if the user request is completely satisfied."
    )

class LangGraphAgent:
    """
    Constructs and compiles the Multi-Agent StateGraph Supervisor architecture.
    """
    def __init__(self, llm_provider, vector_store_provider):
        self.nodes = AgentNodes(llm_provider, vector_store_provider)
        
        # Build the supervisor prompt
        system_prompt = (
            "You are a supervisor managing a conversation between these agents: "
            "rag_agent, search_agent, code_agent, action_agent, writer_agent, critic_agent, mcp_agent.\n"
            "Given the user request and conversation history, respond with the next agent to use.\n\n"
            "AGENT SELECTION RULES (follow strictly):\n"
            "- 'mcp_agent': PRIMARY CHOICE for real-world facts, current events, internet searches, or system info. This agent HAS direct internet access via tools. Use this instead of search_agent.\n"
            "- 'code_agent': Use for ANY math, computation, algorithms, sequences (e.g. fibonacci, prime numbers, calculations, statistics). NEVER use search_agent for math.\n"
            "- 'search_agent': Use ONLY if mcp_agent fails or specifically for standard web searches not covered by MCP tools.\n"
            "- 'rag_agent': Use when the user asks about their own uploaded PDF documents or local files.\n"
            "- 'action_agent': Use when the user wants to DO something externally, like send an email or book a calendar event.\n"
            "- 'writer_agent': Use when the user wants creative writing, formatting, or polishing of text.\n"
            "- 'critic_agent': Use to double-check or review a previous answer for accuracy.\n\n"
            "CRITICAL RULES:\n"
            "1. If an agent returns 'mocked', 'unavailable', or an error, select FINISH immediately.\n"
            "2. DO NOT route to the same agent twice in a row.\n"
            "3. If the request is satisfied, select FINISH."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the conversation above, who should act next? Select one of the agents or FINISH.")
        ])
        
        llm = llm_provider.get_model()
        # Bind the Pydantic tool to enforce structured output
        self.supervisor_chain = prompt | llm.with_structured_output(RouteDecision)
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # --- Keyword pre-router: catches well-known patterns before asking the LLM ---
        MATH_KEYWORDS = ["fibonacci", "prime", "factorial", "calculate", "compute", "square root", "derivative", "integral", "statistics", "mean", "median", "percent", "equation", "formula", "sequence", "series", "algebra"]
        DOC_KEYWORDS = ["document", "pdf", "file", "uploaded", "my files", "according to"]
        WRITE_KEYWORDS = ["poem", "essay", "story", "write me", "creative", "draft", "letter", "format"]
        ACTION_KEYWORDS = ["send email", "email to", "book a", "schedule", "calendar event", "create event"]
        SEARCH_KEYWORDS = ["weather", "news", "today", "current", "latest", "who is", "what is the price", "stock", "live", "time", "system", "os info"]

        def keyword_route(question: str):
            q = question.lower()
            if any(k in q for k in MATH_KEYWORDS):
                return "code_agent"
            if any(k in q for k in DOC_KEYWORDS):
                return "rag_agent"
            if any(k in q for k in WRITE_KEYWORDS):
                return "writer_agent"
            if any(k in q for k in ACTION_KEYWORDS):
                return "action_agent"
            if any(k in q for k in SEARCH_KEYWORDS):
                return "mcp_agent"  # Route to MCP agent for search keywords
            return None  # Fall through to LLM supervisor

        # 1. Define Supervisor Node
        def supervisor_node(state: AgentState):
            # Extract the original user question
            user_question = ""
            for msg in state["messages"]:
                if hasattr(msg, "type") and msg.type == "human":
                    user_question = msg.content
                    break

            # Only run keyword pre-router on first turn (when no agent has responded yet)
            agent_responses = [m for m in state["messages"] if hasattr(m, "name") and m.name]
            if not agent_responses:
                forced_route = keyword_route(user_question)
                if forced_route:
                    print(f"--- SUPERVISOR (keyword rule): Routing to {forced_route} ---")
                    return {"next": forced_route}

            print("--- SUPERVISOR IS THINKING ---")
            decision = self.supervisor_chain.invoke(state)
            return {"next": decision.next}
            
        workflow.add_node("supervisor", supervisor_node)
        
        # 2. Add sub-agent nodes
        workflow.add_node("rag_agent", self.nodes.rag_node)
        workflow.add_node("search_agent", self.nodes.search_node)
        workflow.add_node("code_agent", self.nodes.code_node)
        workflow.add_node("action_agent", self.nodes.action_node)
        workflow.add_node("writer_agent", self.nodes.writer_node)
        workflow.add_node("critic_agent", self.nodes.critic_node)
        workflow.add_node("mcp_agent", self.nodes.mcp_node)
        
        # Always route back back to the supervisor when done
        agents = ["rag_agent", "search_agent", "code_agent", "action_agent", "writer_agent", "critic_agent", "mcp_agent"]
        for agent in agents:
            workflow.add_edge(agent, "supervisor")

        # Create conditional edge from supervisor
        conditional_map = {name: name for name in agents}
        conditional_map["FINISH"] = END
        
        def router_guardrail(state: AgentState):
            next_agent = state["next"]
            if next_agent == "FINISH":
                return "FINISH"
            
            # If the LLM tries to call the exact same agent twice in a row, forcefully END the workflow to prevent loops
            if state["messages"] and hasattr(state["messages"][-1], "name") and state["messages"][-1].name == next_agent:
                print(f"--- GUARDRAIL TRIGGERED: Prevented infinite loop into {next_agent} ---")
                return "FINISH"
            
            # Also prevent endless bouncing back and forth by checking if messages list is getting too long (max 10 iterations)
            if len(state.get("messages", [])) > 10:
                print("--- GUARDRAIL TRIGGERED: Max conversation turns reached. Ending. ---")
                return "FINISH"
                
            return next_agent

        workflow.add_conditional_edges("supervisor", router_guardrail, conditional_map)

        # Build execution connections
        workflow.set_entry_point("supervisor")

        # Compile and return the runnable graph application
        return workflow.compile()
