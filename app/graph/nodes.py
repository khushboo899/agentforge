"""AgentForge graph nodes.

This version keeps the automatic PDF RAG flow and multi-agent routing while adding
Phase 5 observability/reliability: trace events, token accounting, latency, and
provider-error handling.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq

from app.graph.state import AgentState
from app.memory.conversation import add_assistant_message, add_user_message
from app.observability import extract_usage, get_tracker
from app.rag.retriever import retrieve_documents
from app.tools.calculator import calculator

load_dotenv()

MODEL_NAME = os.getenv("AGENTFORGE_MODEL", "openai/gpt-oss-120b")
MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 400
MAX_RAG_DOCUMENTS = 5
MAX_CHARS_PER_DOCUMENT = 1000
MAX_TOTAL_CONTEXT_CHARS = 4500
MAX_CRITIC_ANSWER_CHARS = 2000
DEBUG = os.getenv("AGENTFORGE_DEBUG", "false").lower() == "true"

_llm = None
_groq_client = None


def get_llm():
    """Create the chat model lazily so basic CLI startup doesn't require a key."""
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        _llm = ChatGroq(model=MODEL_NAME, temperature=0)
    return _llm


def get_groq_client():
    """Create the Groq API client only when web research is requested."""
    global _groq_client
    if _groq_client is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        _groq_client = Groq()
    return _groq_client


def debug(message: str) -> None:
    if DEBUG:
        print(f"[AgentForge][DEBUG] {message}")


def _content(response: Any) -> str:
    value = getattr(response, "content", response)
    if isinstance(value, list):
        return "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in value)
    return str(value or "").strip()



def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract the first valid JSON object from an LLM response."""
    if not raw or not raw.strip():
        raise ValueError("The model returned an empty JSON response.")

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()

    # Try each opening brace. This also tolerates text before the JSON.
    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in the model response.")



def call_llm(
    prompt: str,
    max_tokens: int = 400,
    state: AgentState | None = None,
    operation: str = "llm_call",
):
    """Call the model and record usage/latency in the current trace."""
    tracker = get_tracker(state or {})
    try:
        with tracker.span(operation, kind="llm", model=MODEL_NAME):
            response = get_llm().invoke(prompt, max_completion_tokens=max_tokens)
        tracker.llm_usage(MODEL_NAME, extract_usage(response), operation=operation)
        return response
    except Exception as exc:
        tracker.error(operation, exc, recoverable=True)
        raise


def web_research(question: str, state: AgentState) -> str:
    """Use Groq browser_search for genuinely current information."""
    tracker = get_tracker(state)
    try:
        with tracker.span("browser_search", kind="tool", tool="browser_search"):
            response = get_groq_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": (
                        "Research the question using current web information. "
                        "Prefer authoritative recent sources. "
                        "Give a concise factual result and mention important sources. "
                        "If a location abbreviation is ambiguous, do not guess: "
                        "state the possible interpretations and request clarification. "
                        "Distinguish local time zones and dates explicitly. "
                        "Question: " + question[:1200]
                    ),
                }],
                tools=[{"type": "browser_search"}],
                tool_choice="required",
                temperature=0.2,
                max_completion_tokens=350,
            )
        tracker.tool_call("browser_search", operation="web_research")
        usage = extract_usage(response)
        if usage["total_tokens"]:
            tracker.llm_usage(MODEL_NAME, usage, operation="web_research")
        return str(response.choices[0].message.content or "").strip()
    except Exception as exc:
        tracker.error("web_research", exc, recoverable=True)
        raise


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_available_pdfs() -> list[str]:
    """Discover PDFs in the project's data directory, independent of the shell cwd."""
    data_directory = os.path.join(PROJECT_ROOT, "data")
    if not os.path.isdir(data_directory):
        return []
    return sorted(
        os.path.join(data_directory, name)
        for name in os.listdir(data_directory)
        if name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(data_directory, name))
    )


def extract_calculation(question: str) -> str | None:
    """Extract a simple arithmetic expression only from calculation-style requests."""
    q = question.strip()
    match = re.search(
        r"(?:calculate|compute|evaluate|what is|solve)\s*[:=]?\s*([0-9().+*/%\-\s]{1,180})\??$",
        q,
        flags=re.IGNORECASE,
    )
    if match:
        expression = match.group(1).strip()
        if expression and re.search(r"\d", expression):
            return expression
    # A bare arithmetic expression is also an unambiguous calculation request.
    if re.fullmatch(r"[0-9().+*/%\-\s]{1,180}", q) and re.search(r"\d", q):
        return q
    return None


def is_general_query(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        r"^hi$", r"^hello$", r"^hey$", r"^hi there$", r"^hello there$",
        r"^good morning$", r"^good afternoon$", r"^good evening$",
        r"^thanks$", r"^thank you$", r"^thank you so much$",
        r"^who are you\??$", r"^what can you do\??$",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def detect_document_query(question: str) -> bool:
    q = question.lower()
    patterns = [
        r"\bpdf\b", r"\bdocument\b", r"\bfile\b", r"\buploaded\b",
        r"\bthis pdf\b", r"\bthis document\b", r"\bthis file\b",
        r"\bfrom the pdf\b", r"\bin the pdf\b", r"\bfrom pdf\b", r"\bin pdf\b",
        r"\bfrom the document\b", r"\bin the document\b",
        r"\baccording to the pdf\b", r"\baccording to the document\b",
        r"\bsummarize the pdf\b", r"\bsummarise the pdf\b",
        r"\bsummarize the document\b", r"\bsummarise the document\b",
        r"\bextract from the pdf\b", r"\bextract from the document\b",
        r"\bfrom my documents?\b", r"\bmy documents?\b",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def is_live_query(question: str) -> bool:
    q = question.lower()
    patterns = [
        r"\blatest\b", r"\btoday\b", r"\bcurrently\b", r"\bcurrent\b",
        r"\bnow\b", r"\brecent\b", r"\bthis week\b", r"\bthis month\b",
        r"\brecently\b", r"\bup[- ]to[- ]date\b", r"\blive\b", r"\bnews\b",
        r"\bweather\b", r"\bprice\b", r"\bprices\b", r"\bexchange rate\b",
        r"\bstock\b", r"\bmarket today\b", r"\bonline\b",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def looks_like_knowledge_query(question: str) -> bool:
    q = question.lower().strip()
    return bool(re.search(r"^(what|why|how|when|where|explain|define|difference|compare|tell me)", q))


def expand_query(question: str) -> list[str]:
    """Generate cheap deterministic retrieval variants; no LLM call."""
    queries = [question]
    q = question.lower()
    # Common student shorthand: "fx" means function in programming questions.
    if re.search(r"\bfx\b", q):
        queries.extend([
            re.sub(r"\bfx\b", "function", question, flags=re.IGNORECASE),
            "Python function definition parameters arguments return",
        ])
    elif "function" in q or "functions" in q:
        queries.extend([
            "Python function definition parameters arguments return",
            "Python built-in and user-defined functions",
        ])
    if "lambda" in q:
        queries.append("Python lambda function anonymous function")
    return list(dict.fromkeys(queries))[:3]


def memory_node(state: AgentState):
    print("[AgentForge] Loading conversation memory...")
    messages = state.get("messages", [])
    return {"messages": add_user_message(messages=messages, question=state["question"])}


def analyze_task(state: AgentState):
    """Analyze intent cheaply first; use the LLM only for ambiguous queries."""
    question = state["question"]
    print("[AgentForge] Analyzing task...")

    if is_general_query(question):
        return {
            "task_type": "greeting",
            "complexity": "simple",
            "required_tools": [],
            "subtasks": [],
            "requires_rag": False,
        }

    explicit_document = detect_document_query(question)
    live_query = is_live_query(question)
    knowledge_query = looks_like_knowledge_query(question)
    calculation = extract_calculation(question)

    if calculation is not None:
        return {
            "task_type": "calculation",
            "complexity": "simple",
            "required_tools": ["calculator"],
            "subtasks": ["evaluate arithmetic expression safely"],
            "requires_rag": False,
            "calculation_expression": calculation,
        }

    # Deterministic routing is enough for common cases and avoids unnecessary quota use.
    if explicit_document or (knowledge_query and not live_query):
        return {
            "task_type": "question_answering",
            "complexity": "simple",
            "required_tools": [],
            "subtasks": [],
            "requires_rag": True,
        }

    if live_query:
        return {
            "task_type": "current_information",
            "complexity": "medium",
            "required_tools": ["web"],
            "subtasks": ["retrieve current information"],
            "requires_rag": detect_document_query(question),
        }

    # Only ambiguous requests use the classifier LLM.
    prompt = f"""Classify the following task. Return ONLY JSON, no markdown.
Question: {question}
Schema: {{"task_type":"question_answering|explanation|summarization|coding|calculation|general","complexity":"simple|medium|complex","required_tools":[],"subtasks":[],"requires_rag":false}}"""
    try:
        response = call_llm(prompt, 180, state, "task_analysis")
        raw = _content(response)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        raw = re.sub(r"\s*```$", "", raw.strip())
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
    except Exception as exc:
        debug(f"Task analysis fallback: {exc}")
        data = {
            "task_type": "general",
            "complexity": "simple",
            "required_tools": [],
            "subtasks": [],
            "requires_rag": False,
        }

    if explicit_document:
        data["requires_rag"] = True
    return {
        "task_type": data.get("task_type", "general"),
        "complexity": data.get("complexity", "simple"),
        "required_tools": data.get("required_tools", []),
        "subtasks": data.get("subtasks", []),
        "requires_rag": bool(data.get("requires_rag", False)),
    }


def _select_agents(state: AgentState) -> list[str]:
    """Select only agents justified by the request; general prompts go straight to the LLM."""
    question = state["question"]
    if is_general_query(question):
        return []

    agents: list[str] = []
    document_requested = detect_document_query(question)
    live_requested = is_live_query(question)
    task_type = state.get("task_type", "")

    # Explicit document intent or the task analyzer's decision enables local PDF RAG.
    if document_requested or state.get("requires_rag"):
        agents.append("rag")

    # Current-information requests need web retrieval, not an unsupported generic fallback.
    if live_requested or task_type == "current_information" or "web" in state.get("required_tools", []):
        agents.append("web")

    if task_type == "calculation" or "calculator" in state.get("required_tools", []):
        agents.append("calculator")

    return agents


def create_plan(state: AgentState):
    """Build a multi-agent plan without another planner LLM call."""
    question = state["question"]
    print(f"[AgentForge] Planning task: {question}")
    selected = _select_agents(state)

    if not selected:
        plan = ["direct_response"]
    else:
        plan = [f"execute_{agent}" for agent in selected]
        plan.append("synthesize_answer")
        plan.append("critic_review")

    return {
        "plan": plan,
        "selected_agents": selected,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
    }


def rag_node(state: AgentState):
    print("[AgentForge] Starting RAG agent...")
    pdf_paths = get_available_pdfs()
    if not pdf_paths:
        return {"rag_results": [], "evidence": [], "context": "No PDFs are available in data/."}

    queries = expand_query(state["question"])
    print(f"[AgentForge] RAG queries: {queries}")
    collected: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        for query in queries:
            try:
                results = retrieve_documents(question=query, pdf_path=pdf_path)
                collected.extend(results)
            except Exception as exc:
                debug(f"RAG retrieval failed for {pdf_path}: {exc}")

    # Deduplicate and keep the closest results.
    unique: dict[tuple[str, Any, str], dict[str, Any]] = {}
    for item in collected:
        key = (str(item.get("source", "")), item.get("page"), str(item.get("content", ""))[:150])
        unique[key] = item

    ranked = sorted(unique.values(), key=lambda x: float(x.get("score", 9999)))[:MAX_RAG_DOCUMENTS]
    context_parts = []
    for item in ranked:
        source = item.get("source", "unknown")
        page = item.get("page", "?")
        content = str(item.get("content", ""))[:MAX_CHARS_PER_DOCUMENT]
        context_parts.append(f"SOURCE: {source} | PAGE: {page}\n{content}")

    context = "\n\n---\n\n".join(context_parts)[:MAX_TOTAL_CONTEXT_CHARS]
    evidence = [
        {"type": "document", "source": item.get("source"), "page": item.get("page"), "content": item.get("content", ""), "score": item.get("score")}
        for item in ranked
    ]
    print(f"[AgentForge] RAG returned {len(ranked)} documents.")
    return {
        "rag_results": ranked,
        "retrieved_documents": ranked,
        "evidence": evidence,
        "context": context,
    }


def research_node(state: AgentState):
    print(f"[AgentForge] Starting web research: {state['question']}")
    try:
        result = web_research(state["question"], state)
        web_result = {"type": "web", "content": result, "source": "Groq browser search"}
        return {
            "web_results": [web_result],
            "evidence": state.get("evidence", []) + [web_result],
        }
    except Exception as exc:
        print(f"[AgentForge] Web research failed: {exc}")
        return {
            "web_results": [{"type": "web_error", "content": str(exc)}],
            "errors": state.get("errors", []) + [{
                "component": "web_research",
                "type": type(exc).__name__,
                "message": str(exc),
            }],
        }


def tools_node(state: AgentState):
    """Execute the calculator only for deterministic arithmetic requests."""
    expression = state.get("calculation_expression") or extract_calculation(state.get("question", ""))
    if not expression:
        return {
            "code_results": [{"status": "skipped", "message": "No safe arithmetic expression was identified."}],
            "evidence": state.get("evidence", []),
        }
    result = calculator.invoke({"expression": expression})
    tool_result = {
        "status": "ok" if not str(result).startswith("Calculation error:") else "error",
        "tool": "calculator",
        "expression": expression,
        "result": str(result),
    }
    return {
        "code_results": [tool_result],
        "evidence": state.get("evidence", []) + [{"type": "tool", **tool_result}],
        "context": f"Calculator result for {expression}: {result}",
    }


def execute_selected_agents(state: AgentState):
    selected = state.get("selected_agents", [])
    print(f"[AgentForge] Executing agents: {selected}")

    merged = dict(state)
    merged["evidence"] = list(state.get("evidence", []))
    contexts: list[str] = []

    if "rag" in selected:
        print("[AgentForge] → RAG Agent")
        result = rag_node(merged)
        merged.update(result)
        if result.get("context"):
            contexts.append(result["context"])

    if "web" in selected:
        print("[AgentForge] → Web Research Agent")
        result = research_node(merged)
        merged.update(result)
        for item in result.get("web_results", []):
            if item.get("type") == "web" and item.get("content"):
                contexts.append(f"WEB RESULT\n{item['content']}")

    if "calculator" in selected:
        print("[AgentForge] → Calculator Tool")
        result = tools_node(merged)
        merged.update(result)
        if result.get("context"):
            contexts.append(result["context"])

    if "code" in selected:
        print("[AgentForge] → Code Agent")
        result = tools_node(merged)
        merged.update(result)

    if contexts:
        merged["context"] = "\n\n===== AGENT SOURCE =====\n\n".join(contexts)[:7000]

    print("[AgentForge] All selected agents completed.")
    return {
        "rag_results": merged.get("rag_results", []),
        "web_results": merged.get("web_results", []),
        "code_results": merged.get("code_results", []),
        "evidence": merged.get("evidence", []),
        "context": merged.get("context", ""),
        "retrieved_documents": merged.get("retrieved_documents", []),
        "errors": merged.get("errors", []),
    }


def _fallback_answer(state: AgentState) -> str:
    """Produce a useful non-LLM answer when provider quota is unavailable."""
    question = state["question"]
    rag = state.get("rag_results", [])
    web = state.get("web_results", [])

    if rag:
        first = rag[0]
        source = first.get("source", "document")
        page = first.get("page", "?")
        content = str(first.get("content", "")).strip()
        return (
            f"I couldn't use the language model right now, so here is the most relevant "
            f"information retrieved from your documents.\n\n{content}\n\n"
            f"Source: {source}, page {page}."
        )

    if web and web[0].get("content"):
        return str(web[0]["content"])

    return "I couldn't generate the answer because the language-model provider is temporarily unavailable. Please try again after the provider limit resets."


def generate_answer(state: AgentState):
    question = state["question"]
    task_type = state.get("task_type", "general")
    selected = state.get("selected_agents", [])

    if task_type == "greeting":
        return {"answer": "Hello! How can I assist you today?", "generation_fallback": False}

    if task_type == "calculation":
        tool_results = state.get("code_results", [])
        if tool_results:
            item = tool_results[-1]
            if item.get("status") == "ok":
                return {
                    "answer": f"{item.get('expression')} = {item.get('result')}",
                    "generation_fallback": False,
                }
            return {
                "answer": str(item.get("result", "The calculation could not be completed.")),
                "generation_fallback": True,
            }

    # If web was required but failed, never pretend that current information was retrieved.
    web_failed = "web" in selected and not any(
        item.get("type") == "web" and item.get("content")
        for item in state.get("web_results", [])
    )
    rag_available = bool(state.get("rag_results"))

    context = state.get("context", "")

    if web_failed and not rag_available:
        message = (
            "I couldn't retrieve the requested current web information right now. "
            "The web research agent failed, so I won't present unverified current information as fact."
        )
        return {
            "answer": message,
            "generation_fallback": True,
            "generation_error": "web_research_unavailable",
        }

    # Include a short window of prior turns to support follow-up questions.
    history = state.get("messages", [])[:-1]  # memory_node already appended the current user turn
    history = history[-MAX_HISTORY_MESSAGES:]
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:MAX_HISTORY_CHARS]}"
        for item in history
        if isinstance(item, dict) and item.get("content")
    )
    retry_feedback = str(state.get("retry_feedback", "")).strip()

    prompt = f"""You are AgentForge, a multi-agent AI assistant.

Recent conversation:
{history_text or 'No prior conversation.'}

User question:
{question}

Critic feedback from a previous attempt, if any:
{retry_feedback or 'None.'}

Evidence gathered by agents:
{context or 'No external evidence was required or returned.'}

Rules:
- Answer directly and concisely.
- Use document evidence only for claims attributed to the user's documents.
- When using a document chunk, cite its provided source and page, e.g. [python 2.pdf, p. 57].
- Use web evidence only for current/latest claims and name sources actually returned by the web tool.
- If web research failed, explicitly say current web verification was unavailable.
- Do not invent citations, pages, sources, or current facts.
- If evidence is insufficient, say what is missing.
"""

    try:
        response = call_llm(prompt, 500, state, "answer_generation")
        answer = _content(response)
        if not answer:
            raise RuntimeError("The model returned an empty answer.")
        return {"answer": answer, "generation_fallback": False}
    except Exception as exc:
        print(f"[AgentForge] Answer generation failed: {exc}")
        tracker = get_tracker(state)
        tracker.error("answer_generation", exc, recoverable=True)
        fallback = _fallback_answer(state)
        return {
            "answer": fallback,
            "generation_error": str(exc),
            "generation_fallback": True,
            "errors": state.get("errors", []) + [{
                "component": "answer_generation",
                "type": type(exc).__name__,
                "message": str(exc),
            }],
        }



def critic_node(state: AgentState):
    """Review an answer without allowing critic failures to block the workflow."""

    if state.get("generation_fallback"):
        return {
            "approved": True,
            "score": 0.5,
            "critique": {
                "status": "skipped",
                "reason": "Answer used reliability fallback.",
            },
            "issues": [],
            "missing_information": [],
            "suggestions": [],
        }

    prompt = f"""Review this AgentForge answer.

Question:
{state.get("question", "")}

Answer:
{state.get("answer", "")[:MAX_CRITIC_ANSWER_CHARS]}

Evidence:
{state.get("context", "")[:4500]}

Return exactly one valid JSON object with this schema:
{{
  "approved": true,
  "score": 0.8,
  "issues": [],
  "missing_information": [],
  "suggestions": []
}}

Rules:
- approved must be a boolean.
- score must be a number from 0 to 1.
- issues, missing_information, and suggestions must be arrays of strings.
- Do not include Markdown or text outside the JSON.
- Approve a relevant answer supported by available evidence.
"""

    try:
        response = call_llm(prompt, 350, state, "critic")
        data = parse_json_object(_content(response))

        approved = data.get("approved", True)
        score = data.get("score", 0.5)

        if not isinstance(approved, bool):
            raise ValueError("'approved' must be a boolean.")

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("'score' must be numeric.")

        score = max(0.0, min(1.0, float(score)))

        def string_list(key: str) -> list[str]:
            value = data.get(key, [])
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if isinstance(item, str)]

        issues = string_list("issues")
        missing = string_list("missing_information")
        suggestions = string_list("suggestions")

        return {
            "approved": approved,
            "score": score,
            "critique": data,
            "issues": issues,
            "missing_information": missing,
            "suggestions": suggestions,
        }

    except Exception as exc:
        print(
            f"[AgentForge] Critic unavailable; preserving answer "
            f"({type(exc).__name__})."
        )

        tracker = get_tracker(state)
        tracker.error("critic", exc, recoverable=True)

        errors = list(state.get("errors", []))
        errors.append({
            "component": "critic",
            "type": type(exc).__name__,
            "message": str(exc),
        })

        # Fail open: the answer is retained and no critic retry is triggered.
        return {
            "approved": True,
            "score": 0.0,
            "critique": {
                "status": "unavailable",
                "error": str(exc),
            },
            "issues": [],
            "missing_information": [],
            "suggestions": [],
            "errors": errors,
        }



def prepare_retry(state: AgentState):
    retry_count = state.get("retry_count", 0) + 1
    feedback_items = state.get("issues", []) or state.get("suggestions", [])
    feedback = "; ".join(str(item) for item in feedback_items) or "Improve relevance and evidence support."
    print(f"[AgentForge] Replanning after critic feedback (retry {retry_count}).")
    # Remove prior attempt's evidence so a retry cannot accidentally reuse stale agent output.
    return {
        "retry_count": retry_count,
        "retry_feedback": feedback,
        "context": "",
        "rag_results": [],
        "web_results": [],
        "code_results": [],
        "evidence": [],
        "retrieved_documents": [],
        "answer": "",
        "approved": False,
        "generation_error": "",
        "generation_fallback": False,
    }
