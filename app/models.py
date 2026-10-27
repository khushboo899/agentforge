from pydantic import BaseModel, Field


class TaskAnalysis(BaseModel):
    """
    Represents the analysis of a user task.
    """

    task_type: str = Field(
        description="Type of task"
    )

    complexity: str = Field(
        description="Task complexity"
    )

    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools required for the task"
    )

    subtasks: list[str] = Field(
        default_factory=list,
        description="Subtasks needed"
    )

    requires_rag: bool = Field(
        description="Whether document retrieval is required"
    )


class Critique(BaseModel):
    """
    Represents the critic's evaluation
    of the generated answer.
    """

    approved: bool

    score: float

    issues: list[str] = Field(
        default_factory=list
    )

    missing_information: list[str] = Field(
        default_factory=list
    )

    suggestions: list[str] = Field(
        default_factory=list
    )