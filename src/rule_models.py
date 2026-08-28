from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TaskType(Enum):
    training = "training"
    evaluation = "evaluation"


class Claims(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    claim_id: str = Field(alias="id")
    claim: str


class Task(BaseModel):
    task_id: str
    inferred_rule: str
    task_type: TaskType
    reference_claims: list[Claims]
