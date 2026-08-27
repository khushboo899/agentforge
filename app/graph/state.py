from typing import TypedDict, List


class AgentState(TypedDict, total=False):
    question: str

    task_type: str
    complexity: str

    subtasks: List[str]
    required_tools: List[str]
    requires_rag: bool

    plan: List[str]

    route: str

    research_results: List[str]

    answer: str

    critique: str
    approved: bool