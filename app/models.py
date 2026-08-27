from pydantic import BaseModel, Field
from typing import List


class TaskAnalysis(BaseModel):
    task_type: str = Field(
        description="The type of task, such as research, coding, analysis, or general"
    )

    complexity: str = Field(
        description="Task complexity: simple, medium, or complex"
    )

    required_tools: List[str] = Field(
        description="Tools that may be required to solve the task"
    )

    subtasks: List[str] = Field(
        description="Smaller tasks required to complete the main task"
    )

    requires_rag: bool = Field(
        description="Whether document retrieval/RAG is required"
    )