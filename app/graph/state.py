from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):

    # ============================================================
    # USER INPUT
    # ============================================================

    question: str

    messages: List[Dict[str, str]]


    # ============================================================
    # TASK ANALYSIS
    # ============================================================

    task_type: str

    complexity: str

    subtasks: List[str]

    required_tools: List[str]

    plan: List[str]

    selected_agents: List[str]

    requires_rag: bool

    calculation_expression: str


    # ============================================================
    # AGENT OUTPUTS
    # ============================================================

    rag_results: List[Dict[str, Any]]

    web_results: List[Dict[str, Any]]

    code_results: List[Dict[str, Any]]

    evidence: List[Dict[str, Any]]

    context: str

    retrieved_documents: List[Dict[str, Any]]


    # ============================================================
    # FINAL ANSWER
    # ============================================================

    answer: str


    # ============================================================
    # CRITIC
    # ============================================================

    critique: Dict[str, Any]

    approved: bool

    score: float

    issues: List[str]

    missing_information: List[str]

    suggestions: List[str]


    # ============================================================
    # RELIABILITY / RETRY
    # ============================================================

    retry_count: int

    max_retries: int

    generation_error: str

    fatal_error: str

    generation_fallback: bool

    retry_feedback: str

    errors: List[Dict[str, Any]]


    # ============================================================
    # PHASE 5 OBSERVABILITY
    # ============================================================

    trace_id: str

    trace_summary: Dict[str, Any]