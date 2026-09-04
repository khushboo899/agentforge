from app.models import TaskAnalysis
from app.graph.state import AgentState
from app.rag.retriever import retrieve_documents

from app.memory.conversation import (
    add_user_message,
    add_assistant_message
)

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def analyze_task(state: AgentState):

    print("\n🔍 Analyzing task...")

    prompt = f"""
You are a task analysis agent.

Analyze the user's request and return ONLY valid JSON.

The JSON must follow this exact structure:

{{
    "task_type": "string",
    "complexity": "simple | medium | complex",
    "required_tools": ["tool1", "tool2"],
    "subtasks": ["subtask1", "subtask2"],
    "requires_rag": true
}}

Rules:

- task_type must describe the task.
- complexity must be simple, medium, or complex.
- required_tools must be a JSON array.
- subtasks must be a JSON array.
- requires_rag must be true or false.
- Do not return markdown.
- Do not return explanations.
- Return JSON only.

USER REQUEST:
{state["question"]}
"""

    response = llm.invoke(prompt)

    import json

    try:
        data = json.loads(response.content)

    except json.JSONDecodeError:

        print("\n❌ Invalid JSON returned by model.")
        print(response.content)

        raise ValueError(
            "Task analyzer returned invalid JSON."
        )

    return {
        "task_type": data["task_type"],
        "complexity": data["complexity"],
        "subtasks": data["subtasks"],
        "required_tools": data["required_tools"],
        "requires_rag": data["requires_rag"],
    }


def create_plan(state: AgentState):

    print("\n🧠 Creating plan...")

    plan = []

    for i, subtask in enumerate(
        state.get("subtasks", []),
        start=1
    ):
        plan.append(f"Step {i}: {subtask}")

    # Add critic feedback during retries
    missing_information = state.get(
        "missing_information",
        []
    )

    suggestions = state.get(
        "suggestions",
        []
    )

    if missing_information:

        print("\n⚠️ Adding missing information to plan...")

        for item in missing_information:
            plan.append(
                f"Additional research: {item}"
            )

    if suggestions:

        for suggestion in suggestions:
            plan.append(
                f"Improvement: {suggestion}"
            )

    return {
        "plan": plan
    }


def route_task(state: AgentState):

    print("\n🚦 Routing task...")

    # Rule 1: If a document is provided, use RAG
    if state.get("pdf_path"):
        route = "rag"

    # Rule 2: Use RAG if task analysis explicitly requires it
    elif state.get("requires_rag"):
        route = "rag"

    # Rule 3: Use tools if tools are explicitly required
    elif state.get("required_tools"):
        route = "tools"

    # Default
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

    documents = retrieve_documents(
        question=state["question"],
        pdf_path=state["pdf_path"]
    )

    # No relevant documents found
    if not documents:

        return {
            "retrieved_documents": [],
            "context": "NO_RELEVANT_CONTEXT_FOUND"
        }

    context_parts = []

    for doc in documents:

        chunk_text = (
            f"Source: {doc['source']}\n"
            f"Page: {doc['page']}\n"
            f"Content: {doc['content']}"
        )

        context_parts.append(chunk_text)

    context = "\n\n---\n\n".join(
        context_parts
    )

    return {
        "retrieved_documents": documents,
        "context": context
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
    
    messages = state.get(
    "messages",
    []
)
    
    conversation_history = "\n".join(
    f"{message['role'].upper()}: {message['content']}"
    for message in messages
)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the provided context.

IMPORTANT RULES:

1. Use information from the provided context when available.
2. Do not invent information not supported by the context.
3. If the context is NO_RELEVANT_CONTEXT_FOUND,
   clearly state that the provided document does not contain
   sufficient information to answer the question.
4. Do not use outside knowledge when answering RAG questions.
5. When using information from a source, mention page numbers.
6. At the end, provide a Sources section listing relevant pages.

User Question:
{state["question"]}

Retrieved Context:
{state.get("context", "")}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def critic_node(state: AgentState):

    print("\n🧐 Critic evaluating answer...")

    prompt = f"""
You are a strict quality-control agent.

Evaluate the AI-generated answer.

USER QUESTION:
{state["question"]}

CURRENT ANSWER:
{state["answer"]}

RESEARCH RESULTS:
{state.get("research_results", [])}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "approved": true,
    "score": 0.85,
    "issues": [],
    "missing_information": [],
    "suggestions": []
}}

Rules:

- approved must be true or false.
- score must be between 0.0 and 1.0.
- issues must be a JSON array of strings.
- missing_information must be a JSON array of strings.
- suggestions must be a JSON array of strings.
- Approve only if the answer is sufficiently complete,
  relevant, and useful.
- Do not return markdown.
- Do not return explanations.
- Return JSON only.
"""

    response = llm.invoke(prompt)

    import json

    try:
        data = json.loads(response.content)

    except json.JSONDecodeError:

        print("\n❌ Critic returned invalid JSON:")
        print(response.content)

        raise ValueError(
            "Critic returned invalid JSON."
        )

    print(f"Critic score: {data['score']}")
    print(f"Approved: {data['approved']}")

    return {
        "critique": data,
        "approved": data["approved"],
        "score": data["score"],
        "issues": data["issues"],
        "missing_information": data["missing_information"],
        "suggestions": data["suggestions"],
    } 
    
def prepare_retry(state: AgentState):

    retry_count = state.get("retry_count", 0)

    retry_count += 1

    print(f"\n🔄 Retry attempt: {retry_count}")

    return {
        "retry_count": retry_count
    }   

def memory_node(state: AgentState):

    print("\n🧠 Loading conversation memory...")

    messages = state.get(
        "messages",
        []
    )

    updated_messages = add_user_message(
        messages=messages,
        question=state["question"]
    )

    return {
        "messages": updated_messages
    } 