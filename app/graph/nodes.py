from app.models import TaskAnalysis
from app.graph.state import AgentState

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def analyze_task(state: AgentState):

    print("\n🔍 Analyzing task...")

    structured_llm = llm.with_structured_output(TaskAnalysis)

    result = structured_llm.invoke(state["question"])

    return {
        "task_type": result.task_type,
        "complexity": result.complexity,
        "subtasks": result.subtasks,
        "required_tools": result.required_tools,
        "requires_rag": result.requires_rag,
    }


def create_plan(state: AgentState):

    print("\n🧠 Creating plan...")

    plan = []

    for i, subtask in enumerate(state["subtasks"], start=1):
        plan.append(f"Step {i}: {subtask}")

    return {
        "plan": plan
    }


def route_task(state: AgentState):

    print("\n🚦 Routing task...")

    if state.get("requires_rag"):
        route = "rag"

    elif state.get("required_tools"):
        route = "tools"

    else:
        route = "research"

    print(f"Route selected: {route}")

    return {
        "route": route
    }


def research_node(state: AgentState):

    print("\n🔎 Research agent running...")

    results = [
        "Research would be performed here.",
        "External information would be gathered here."
    ]

    return {
        "research_results": results
    }


def rag_node(state: AgentState):

    print("\n📚 RAG agent running...")

    results = [
        "Relevant documents would be retrieved here.",
        "Retrieved context would be passed to the LLM."
    ]

    return {
        "research_results": results
    }


def tools_node(state: AgentState):

    print("\n🔧 Tool agent running...")

    results = [
        "Required tools would be executed here.",
        "Tool results would be added to the state."
    ]

    return {
        "research_results": results
    }


def generate_answer(state: AgentState):

    print("\n✍️ Generating answer...")

    prompt = f"""
You are an AI decision-making assistant.

User question:
{state["question"]}

Plan:
{state["plan"]}

Research results:
{state.get("research_results", [])}

Generate a clear and useful answer.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def critic_node(state: AgentState):

    print("\n🧐 Critic evaluating answer...")

    prompt = f"""
Evaluate the following AI answer.

Question:
{state["question"]}

Answer:
{state["answer"]}

Return only one word:

APPROVED

or

REJECTED
"""

    response = llm.invoke(prompt)

    approved = "APPROVED" in response.content.upper()

    print("Critic:", response.content)

    return {
        "critique": response.content,
        "approved": approved
    }