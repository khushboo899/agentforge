from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState

from app.graph.nodes import (
    analyze_task,
    create_plan,
    route_task,
    research_node,
    rag_node,
    tools_node,
    generate_answer,
    critic_node,
)


def route_after_planner(state: AgentState):

    return state["route"]


def route_after_critic(state: AgentState):

    if state["approved"]:
        return "end"

    return "retry"


builder = StateGraph(AgentState)


# Nodes
builder.add_node("analyze", analyze_task)
builder.add_node("planner", create_plan)
builder.add_node("router", route_task)

builder.add_node("research", research_node)
builder.add_node("rag", rag_node)
builder.add_node("tools", tools_node)

builder.add_node("generate", generate_answer)
builder.add_node("critic", critic_node)


# Starting point
builder.add_edge(START, "analyze")

# Main pipeline
builder.add_edge("analyze", "planner")
builder.add_edge("planner", "router")


# Conditional routing
builder.add_conditional_edges(
    "router",
    route_after_planner,
    {
        "research": "research",
        "rag": "rag",
        "tools": "tools",
    }
)


# All paths eventually generate an answer
builder.add_edge("research", "generate")
builder.add_edge("rag", "generate")
builder.add_edge("tools", "generate")


# Critic
builder.add_edge("generate", "critic")


# Critic decision
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "end": END,
        "retry": "planner",
    }
)


graph = builder.compile()