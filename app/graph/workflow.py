import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState

from app.graph.nodes import (
    memory_node,
    analyze_task,
    create_plan,
    route_task,
    research_node,
    rag_node,
    tools_node,
    generate_answer,
    critic_node,
    prepare_retry
)



def route_after_planner(state: AgentState):

    return state["route"]


def route_after_critic(state: AgentState):

    score = state.get("score", 0.0)
    approved = state.get("approved", False)
    retry_count = state.get("retry_count", 0)

    print("\n🔀 Critic decision...")

    print(f"   Score: {score}")
    print(f"   Approved: {approved}")
    print(f"   Retry count: {retry_count}")

    # Answer meets quality threshold
    if approved and score >= 0.75:

        print("   ✅ Answer approved.")

        return "end"

    # Prevent infinite retries
    if retry_count >= 2:

        print(
            "   ⚠️ Maximum retries reached. "
            "Returning best available answer."
        )

        return "end"

    # Answer needs improvement
    print("   🔄 Answer needs improvement.")

    return "retry"


builder = StateGraph(AgentState)


# Nodes
builder.add_node("memory", memory_node)
builder.add_node("analyze", analyze_task)
builder.add_node("planner", create_plan)
builder.add_node("router", route_task)

builder.add_node("research", research_node)
builder.add_node("rag", rag_node)
builder.add_node("tools", tools_node)

builder.add_node("generate", generate_answer)
builder.add_node("critic", critic_node)

builder.add_node("prepare_retry", prepare_retry)


# Starting point
builder.add_edge(START, "memory")
builder.add_edge("memory", "analyze")

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
        "retry": "prepare_retry",
    }
)

builder.add_edge("prepare_retry","planner")

conn = sqlite3.connect(
    "checkpoints.sqlite",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

graph = builder.compile(
    checkpointer=checkpointer
)