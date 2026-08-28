"""Named JSON representation of the categorical high-level actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from src.datasets.actions import ActionVector, HighLevelActionEvent


class Phase(StrEnum):
    GRASP = "grasp"
    PLACE = "place"


class LocationType(StrEnum):
    RESERVE = "reserve"
    OUTPUT = "output"


class VLMAction(BaseModel):
    """One categorical simulator action with explicit field names."""

    model_config = ConfigDict(extra="forbid", strict=True)

    phase: Phase
    color_id: int = Field(ge=1, le=9)
    location_type: LocationType
    row: int
    column: int

    def as_vector(self) -> ActionVector:
        return (
            1 if self.phase is Phase.GRASP else 2,
            self.color_id,
            0 if self.location_type is LocationType.RESERVE else 1,
            self.row,
            self.column,
        )


class ActionResponseBase(BaseModel):
    """Base class for VLM responses."""

    model_config = ConfigDict(extra="forbid", strict=True)
    actions: list[VLMAction]


class GridResponseBase(BaseModel):
    """Base class for VLM responses."""

    model_config = ConfigDict(extra="forbid", strict=True)
    output_grid: list[list[int]]


class ActionResponseWithSceneDescriptions(BaseModel):
    """Action response with a compact visual description scaffold."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scene_descriptions: str = Field(min_length=1, max_length=4000)
    actions: list[VLMAction]


class GridResponseWithSceneDescriptions(BaseModel):
    """Grid response with a compact visual description scaffold."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scene_descriptions: str = Field(min_length=1, max_length=4000)
    output_grid: list[list[int]]


class ActionResponse(BaseModel):
    """Strict top-level response returned by the VLM."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rule_summary: str = Field(
        min_length=1,
        max_length=500,
        description="Concise textual statement of the inferred shared transformation.",
    )
    actions: list[VLMAction]


class GridResponse(BaseModel):
    """Strict response returned when directly predicting the output grid."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rule_summary: str = Field(min_length=1, max_length=500)
    output_grid: list[list[int]]


class ActionResponseWithRuleAndSceneDescriptions(BaseModel):
    """Action response with both the requested description and rule fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scene_descriptions: str = Field(min_length=1, max_length=4000)
    rule_summary: str = Field(min_length=1, max_length=500)
    actions: list[VLMAction]


class GridResponseWithRuleAndSceneDescriptions(BaseModel):
    """Grid response with both the requested description and rule fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scene_descriptions: str = Field(min_length=1, max_length=4000)
    rule_summary: str = Field(min_length=1, max_length=500)
    output_grid: list[list[int]]


def parse_action_response(
    response: str,
) -> tuple[
    ActionResponse
    | ActionResponseBase
    | ActionResponseWithSceneDescriptions
    | ActionResponseWithRuleAndSceneDescriptions
    | None,
    str | None,
]:
    """Parse a VLM response into an ActionResponse object or ActionResponseBase in case no rule_summary."""
    try:
        if "rule_summary" in response and "scene_descriptions" in response:
            return ActionResponseWithRuleAndSceneDescriptions.model_validate_json(
                response
            ), None
        if "rule_summary" in response:
            return ActionResponse.model_validate_json(response), None
        if "scene_descriptions" in response:
            return ActionResponseWithSceneDescriptions.model_validate_json(
                response
            ), None
        return ActionResponseBase.model_validate_json(response), None
    except ValidationError as error:
        return None, str(error)


def parse_grid_response(
    response: str,
) -> tuple[
    GridResponse
    | GridResponseBase
    | GridResponseWithSceneDescriptions
    | GridResponseWithRuleAndSceneDescriptions
    | None,
    str | None,
]:
    """Parse a VLM response into a GridResponse object or GridResponseBase in case no rule_summary."""
    try:
        if "rule_summary" in response and "scene_descriptions" in response:
            return GridResponseWithRuleAndSceneDescriptions.model_validate_json(
                response
            ), None
        if "rule_summary" in response:
            return GridResponse.model_validate_json(response), None
        if "scene_descriptions" in response:
            return GridResponseWithSceneDescriptions.model_validate_json(response), None
        return GridResponseBase.model_validate_json(response), None
    except ValidationError as error:
        return None, str(error)


def action_from_vector(action: ActionVector) -> VLMAction:
    phase, color_id, location_type, row, column = action
    return VLMAction(
        phase=Phase.GRASP if phase == 1 else Phase.PLACE,
        color_id=color_id,
        location_type=(
            LocationType.RESERVE if location_type == 0 else LocationType.OUTPUT
        ),
        row=row,
        column=column,
    )


def action_from_event(event: HighLevelActionEvent) -> VLMAction:
    return action_from_vector(event.action)
