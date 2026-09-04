from typing import TypedDict, List


class AgentState(TypedDict, total=False):

    # User input
    question: str
    pdf_path: str

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

    # Final answer
    answer: str

    # Critic
    critique: str
    approved: bool
    score: float
    issues: List[str]
    missing_information: List[str]
    suggestions: List[str]

    # Retry control
    retry_count: int