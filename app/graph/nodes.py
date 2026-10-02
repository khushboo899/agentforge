from app.models import TaskAnalysis , Critique
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

- Use requires_rag = true ONLY when the user question
  requires information from an uploaded/provided document.

- Questions asking about current, recent, latest, today's,
  trending, news, market conditions, live information,
  or recent developments should NOT use RAG.

- Such current-information questions should require a
  research/web tool.

- If no document is provided, do not set requires_rag to true
  merely because the question needs external information.

Examples:

"Summarize the uploaded PDF"
requires_rag = true

"What does page 10 of the PDF say?"
requires_rag = true

"What are today's AI trends?"
requires_rag = false
required_tools = ["web_search"]

"What is the latest OpenAI news?"
requires_rag = false
required_tools = ["web_search"]

"Calculate 125 * 48"
requires_rag = false
required_tools = ["calculator"]

"Explain recursion"
requires_rag = false

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

    # Original subtasks
    for i, subtask in enumerate(
        state.get("subtasks", []),
        start=1
    ):
        plan.append(f"Step {i}: {subtask}")

    # Critic feedback from previous attempt
    issues = state.get("issues", [])
    missing_information = state.get(
        "missing_information",
        []
    )
    suggestions = state.get(
        "suggestions",
        []
    )

    # Add issues
    if issues:

        print("\n⚠️ Addressing critic issues...")

        for issue in issues:

            plan.append(
                f"Fix issue: {issue}"
            )

    # Add missing information
    if missing_information:

        print("\n📌 Adding missing information...")

        for item in missing_information:

            plan.append(
                f"Additional information required: {item}"
            )

    # Add suggestions
    if suggestions:

        print("\n💡 Applying critic suggestions...")

        for suggestion in suggestions:

            plan.append(
                f"Improvement: {suggestion}"
            )

    print("\n📋 Final plan:")

    for step in plan:
        print(f" - {step}")

    return {
        "plan": plan
    }


def route_task(state: AgentState):

    print("\n🚦 Routing task...")

    pdf_path = state.get("pdf_path")
    requires_rag = state.get("requires_rag", False)
    required_tools = state.get("required_tools", [])

    # Rule 1: RAG only when a PDF/document actually exists
    if pdf_path:
        route = "rag"

    # Rule 2: Tools explicitly required
    elif required_tools:
        route = "tools"

    # Rule 3: RAG requested but no document exists
    elif requires_rag:
        print(
            "⚠️ RAG requested but no PDF was provided."
        )

        route = "research"

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

    pdf_path = state.get("pdf_path")

    if not pdf_path:

        print("❌ No PDF provided for RAG.")

        return {
            "retrieved_documents": [],
            "context": "NO_DOCUMENT_PROVIDED"
        }

    documents = retrieve_documents(
        question=state["question"],
        pdf_path=pdf_path
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

Answer the user's question using the provided context
and relevant conversation history.

IMPORTANT RULES:

1. Use information from the provided context when available.
2. Do not invent information not supported by the context.
3. If the context is NO_RELEVANT_CONTEXT_FOUND,
   clearly state that the provided document does not contain
   sufficient information to answer the question.
4. Do not use outside knowledge when answering RAG questions.
5. When using information from a source, mention page numbers.
6. At the end, provide a Sources section listing relevant pages.
7. Use conversation history only when it is relevant to the
   current question.
8. Do not repeat unnecessary information from previous turns.

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER QUESTION:
{state["question"]}

RETRIEVED CONTEXT:
{state.get("context", "")}
"""

    response = llm.invoke(prompt)

    answer = response.content

    messages = state.get(
    "messages",
    []
    )

    updated_messages = add_assistant_message(
        messages=messages,
        answer=answer
    )

    return {
    "answer": answer,
    "messages": updated_messages
    }


def critic_node(state: AgentState):
    print("\n🧐 Critic evaluating answer...")

    question = state.get("question", "")
    answer = state.get("answer", "")
    context = state.get("context", "")
    research_results = state.get("research_results", [])

    prompt = f"""
You are a strict quality-control agent.

Your job is to evaluate the AI-generated answer against the user's
question and the available evidence.

USER QUESTION:
{question}

CURRENT ANSWER:
{answer}

AVAILABLE DOCUMENT CONTEXT:
{context}

RESEARCH / TOOL RESULTS:
{research_results}

Evaluate the answer based on:

1. Correctness
2. Relevance to the user's question
3. Completeness
4. Clarity
5. Whether important information is missing
6. Whether the answer is supported by the available context
7. Whether the answer follows the task requirements

For RAG/document-based questions:
- Check whether the answer is supported by the retrieved context.
- Penalize unsupported claims.
- Check whether important source/page information is missing.

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
- Approve only when the answer is sufficiently correct,
  complete, relevant, and useful.
- Do not return markdown.
- Do not return explanations.
- Return JSON only.
"""

    response = llm.invoke(prompt)

    import json

    raw_content = response.content.strip()

    # Remove markdown code fences if the model accidentally adds them
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]

    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]

    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    raw_content = raw_content.strip()

    try:
        data = json.loads(raw_content)

        # Validate using Pydantic
        critique = Critique.model_validate(data)

    except (json.JSONDecodeError, ValueError) as e:

        print("\n❌ Critic returned invalid output:")
        print(response.content)

        raise ValueError(
            f"Critic returned invalid structured output: {e}"
        )

    print(f"\n📊 Critic score: {critique.score}")
    print(f"✅ Approved: {critique.approved}")

    if critique.issues:
        print("\n⚠️ Issues:")
        for issue in critique.issues:
            print(f" - {issue}")

    if critique.missing_information:
        print("\n📌 Missing information:")
        for item in critique.missing_information:
            print(f" - {item}")

    if critique.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in critique.suggestions:
            print(f" - {suggestion}")

    return {
        "critique": critique.model_dump(),
        "approved": critique.approved,
        "score": critique.score,
        "issues": critique.issues,
        "missing_information": critique.missing_information,
        "suggestions": critique.suggestions,
    }  
    

def prepare_retry(state: AgentState):

    print("\n🔄 Preparing retry...")

    retry_count = state.get("retry_count", 0) + 1

    print(f"   Retry attempt: {retry_count}")

    return {
        "retry_count": retry_count,
        "answer": ""
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