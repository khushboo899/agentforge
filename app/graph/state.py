from typing import TypedDict, List, Dict


class AgentState(TypedDict, total=False):

    # User input
    question: str
    pdf_path: str

    # Conversation memory
    messages: List[Dict[str, str]]

    # Task analysis
    task_type: str
    complexity: str
    subtasks: List[str]
    required_tools: List[str]
    requires_rag: bool

    # Planning
    plan: List[str]

    # Routing
    route: str

    # Research
    research_results: List[str]

    # RAG
    retrieved_documents: List[Dict]
    context: str

    # Final answer
    answer: str

    # Critic
    critique: Dict
    approved: bool
    score: float
    issues: List[str]
    missing_information: List[str]
    suggestions: List[str]

    # Retry
    retry_count: int