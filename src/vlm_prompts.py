"""Composable prompt construction for ERA Bench evaluations."""

from __future__ import annotations

import json
from typing import Any

from src.datasets import HighLevelActionEvent
from src.vlm_actions import (
    ActionResponse,
    ActionResponseBase,
    ActionResponseWithRuleAndSceneDescriptions,
    ActionResponseWithSceneDescriptions,
    GridResponse,
    GridResponseBase,
    GridResponseWithRuleAndSceneDescriptions,
    GridResponseWithSceneDescriptions,
    action_from_event,
)
from src.vlm_preprocessing import VLMDemonstration, VLMTaskInput, describe_action

EMBODIED_CORE_PROMPT = (
    "You solve embodied ARC tasks from solved robot demonstrations.\n"
)

TEXTUAL_CORE_PROMPT = """You solve ARC tasks from solved grid demonstrations.

Infer the single abstract transformation shared by the demonstrations, apply it
to the test input, and return the complete newly derived output grid.
"""

ACTION_CONTEXT_PROMPT = """Every action is a JSON object with the categorical fields phase, color_id,
location_type, row, and column.

Action semantics:
- A reserve action uses location_type "reserve", row -1, and column -1.
- An output-grid action uses location_type "output" and a zero-based output
  row/column.
- To correct an output cell, grasp its cube from the output grid, place that
  cube in the reserve, then grasp the replacement cube and place it in output.
- ARC colors are: 0 black/background, 1 blue, 2 red, 3 green, 4 yellow,
  5 gray, 6 magenta, 7 orange, 8 teal, and 9 brown.
- Black/background cells are empty and require no cube.
- To create a colored output cell, grasp that color from the reserve and place
  it at the required output coordinate. Do not place onto an occupied output
  coordinate; remove its current cube first.
- Return one object per semantic action, without real-time repetitions.
- An entirely black output is represented by an empty action list.
"""

GRID_CONTEXT_PROMPT = """Each textual demonstration contains an input_grid and
the corresponding output_grid. Compare every pair to infer one transformation,
then apply it to the test input_grid. Each grid is rectangular. Grid values are
ARC color IDs: 0 black/background, 1 blue, 2 red, 3 green, 4 yellow, 5 gray,
6 magenta, 7 orange, 8 teal, and 9 brown. Preserve the complete rectangular
output grid, including black/background cells.
"""

RENDERED_SCENE_BACKGROUND_PROMPT = """The input and output grid areas are marked by fixed white lattice borders.
A dark gray or black cell interior with no colored raised cube is an empty ARC
cell (color 0), not a black cube. Only a distinct colored cube inside a lattice
cell makes that cell occupied.

The reserve region has a fixed purple rectangular background visible between
the reserve-cube columns. That purple surface is not an ARC grid, cell, cube, or
task color and contains no rule information. Only the distinct raised colored
cubes beside it are reserve cubes.
"""

FRAME_CONTEXT_PROMPT = f"""Each demonstration is an ordered sequence of sampled `table_cam` RGB frames. The scene shows
an ARC input grid on the left, a robot and reserve cubes in the middle, and an
output grid on the right. Infer the single abstract transformation shared by the
demonstrations, then produce a new action plan for the test input's output.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The reserve area is only a supply of cubes used to physically construct the
output. The reserve cubes' positions, stack order, quantities, and arrangement
contain no information about the ARC transformation. The robot motion and
actions show how a demonstration output was constructed; they do not determine
why that output is correct. First derive the target output solely by applying
the shared transformation to the test input. Only then translate that target
output into robot actions using cubes from the reserve.

All input and output grids are centered on fixed world-space center points. Their
top-left corners therefore move when their dimensions change. Rows and columns
in actions are logical coordinates relative to the inferred output grid, not
fixed pixel or world coordinates: (0, 0) is always that grid's top-left cell.
"""

GRID_FRAME_CONTEXT_PROMPT = f"""Each demonstration is an ordered sequence of sampled `table_cam` RGB frames. The scene shows
an ARC input grid on the left, a robot and reserve cubes in the middle, and an
output grid on the right. Infer the single abstract transformation shared by the
demonstrations, then derive the test input's new output grid.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The reserve area is only a supply of cubes used to physically construct the
output. Its positions, stack order, quantities, and arrangement contain no
information about the ARC transformation. Robot motion shows how a demonstration
output was physically constructed; it does not determine why that output is
correct. Derive the target output solely by applying the shared transformation
to the test input.

All input and output grids are centered on fixed world-space center points. Their
top-left corners therefore move when their dimensions change.
"""

FINAL_FRAME_CONTEXT_PROMPT = f"""Each demonstration provides one final RGB image from each selected camera perspective.
The scene shows an ARC input grid on the left, a robot and reserve cubes in the
middle, and an output grid on the right. Infer the single abstract
transformation shared by the demonstrations, then produce a new action plan for
the test input's output.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The reserve area is only a supply of cubes used to physically construct the
output. The reserve cubes' positions, stack order, quantities, and arrangement
contain no information about the ARC transformation. The initial and final
states show the result of construction, not why that result is correct. First
derive the target output solely by applying the shared transformation to the
test input. Only then translate that target output into robot actions using
cubes from the reserve.

All input and output grids are centered on fixed world-space center points. Their
top-left corners therefore move when their dimensions change. Rows and columns
in actions are logical coordinates relative to the inferred output grid, not
fixed pixel or world coordinates: (0, 0) is always that grid's top-left cell.
"""

GRID_FINAL_FRAME_CONTEXT_PROMPT = f"""Each demonstration provides one final RGB image from each selected camera perspective.
The scene shows an ARC input grid on the left, a robot and reserve cubes in the
middle, and an output grid on the right. Infer the single abstract
transformation shared by the demonstrations, then derive the test input's new
output grid.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The reserve area is only a supply of cubes used to physically construct the
output. Its positions, stack order, quantities, and arrangement contain no
information about the ARC transformation. The initial and final states show the
result of construction, not why that result is correct. Derive the target
output solely by applying the shared transformation to the test input.

All input and output grids are centered on fixed world-space center points. Their
top-left corners therefore move when their dimensions change.
"""

TRANSFORMATION_PROMPT = """The demonstrations are examples, not a plan to copy. Apply their shared rule to
the test input and generate the actions for its new output."""


def _action_output_schema(
    induce_rule_summary: bool, induce_scene_descriptions: bool
) -> dict[str, Any]:
    if induce_rule_summary and induce_scene_descriptions:
        return ActionResponseWithRuleAndSceneDescriptions.model_json_schema()
    if induce_rule_summary:
        return ActionResponse.model_json_schema()
    if induce_scene_descriptions:
        return ActionResponseWithSceneDescriptions.model_json_schema()
    return ActionResponseBase.model_json_schema()


def _grid_output_schema(
    induce_rule_summary: bool, induce_scene_descriptions: bool
) -> dict[str, Any]:
    if induce_rule_summary and induce_scene_descriptions:
        return GridResponseWithRuleAndSceneDescriptions.model_json_schema()
    if induce_rule_summary:
        return GridResponse.model_json_schema()
    if induce_scene_descriptions:
        return GridResponseWithSceneDescriptions.model_json_schema()
    return GridResponseBase.model_json_schema()


def response_schema_for_output_mode(
    output_mode: str,
    induce_rule_summary: bool,
    induce_scene_descriptions: bool = False,
) -> dict[str, Any]:
    return {
        "actions": _action_output_schema(
            induce_rule_summary, induce_scene_descriptions
        ),
        "grid": _grid_output_schema(induce_rule_summary, induce_scene_descriptions),
    }[output_mode]


def build_vlm_messages(
    task: VLMTaskInput,
    *,
    output_mode: str,
    include_frames: bool,
    include_textual_grids: bool,
    include_actions: bool,
    induce_rule_summary: bool,
    induce_scene_descriptions: bool = False,
    include_final_frames: bool = False,
    demonstration_video_urls: list[str] | None = None,
    demonstration_final_image_urls: list[list[tuple[str, str]]] | None = None,
    test_image_url: list[tuple[str, str]] | str | None = None,
) -> list[dict[str, Any]]:
    """Assemble a prompt from stable output, action, grid, and frame sections."""
    _validate_prompt_inputs(
        task,
        output_mode=output_mode,
        include_frames=include_frames,
        include_textual_grids=include_textual_grids,
        include_actions=include_actions,
        include_final_frames=include_final_frames,
        demonstration_video_urls=demonstration_video_urls,
        demonstration_final_image_urls=demonstration_final_image_urls,
        test_image_url=test_image_url,
    )

    content: list[dict[str, Any]] = []
    video_urls = demonstration_video_urls or []
    final_image_urls = demonstration_final_image_urls or []
    test_image_urls = _normalize_camera_images(test_image_url)
    for index, demonstration in enumerate(task.demonstrations, start=1):
        content.append(
            {
                "type": "text",
                "text": _demonstration_heading(
                    index,
                    demonstration,
                    include_frames=include_frames,
                    include_final_frames=include_final_frames,
                ),
            }
        )
        if include_frames:
            content.append(_video_part(video_urls[index - 1]))
        if include_final_frames:
            for camera, image_url in final_image_urls[index - 1]:
                content.append(
                    {
                        "type": "text",
                        "text": f"Final frame — camera perspective: {camera}",
                    }
                )
                content.append(_image_part(image_url))
        if include_textual_grids:
            content.append(
                {"type": "text", "text": format_demonstration_grids(demonstration)}
            )
        if include_actions:
            content.append(
                {
                    "type": "text",
                    "text": format_demonstration_actions(demonstration.actions),
                }
            )

    content.append(
        {
            "type": "text",
            "text": _test_heading(
                task.test_input_grid_size,
                include_visual_context=include_frames or include_final_frames,
            ),
        }
    )
    if include_frames or include_final_frames:
        for camera, image_url in test_image_urls:
            if len(test_image_urls) > 1 or camera != "table_cam":
                content.append(
                    {"type": "text", "text": f"Camera perspective: {camera}"}
                )
            content.append(_image_part(image_url))
    if include_textual_grids:
        content.append(
            {"type": "text", "text": format_test_input_grid(task.test_input_grid)}
        )
    content.append(
        {
            "type": "text",
            "text": _task_instruction(
                output_mode,
                induce_rule_summary,
                include_frames,
                include_actions,
                include_textual_grids,
                include_final_frames,
                induce_scene_descriptions,
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": _system_prompt(
                output_mode=output_mode,
                include_frames=include_frames,
                include_textual_grids=include_textual_grids,
                include_actions=include_actions,
                induce_rule_summary=induce_rule_summary,
                induce_scene_descriptions=induce_scene_descriptions,
                include_final_frames=include_final_frames,
            ),
        },
        {"role": "user", "content": content},
    ]


def format_demonstration_actions(events: list[HighLevelActionEvent]) -> str:
    """Show named categorical actions alongside their short explanations."""
    if not events:
        return "Compressed high-level actions: [] (no colored output cubes)."
    lines = ["Compressed high-level actions, in execution order:"]
    for index, event in enumerate(events, start=1):
        action_json = json.dumps(action_from_event(event).model_dump(mode="json"))
        lines.append(f"{index}. {action_json} — {describe_action(event)}")
    return "\n".join(lines)


def format_demonstration_grids(demonstration: VLMDemonstration) -> str:
    """Render the solved ARC grid pair already loaded for a demonstration."""
    return (
        "Demonstration ARC grids:\n"
        f"input_grid: {json.dumps(demonstration.input_grid)}\n"
        f"output_grid: {json.dumps(demonstration.output_grid)}"
    )


def format_test_input_grid(input_grid: list[list[int]] | None) -> str:
    """Render the held-out input grid without exposing its target grid."""
    return f"Test ARC input grid:\ninput_grid: {json.dumps(input_grid)}"


def _system_prompt(
    *,
    output_mode: str,
    include_frames: bool,
    include_textual_grids: bool,
    include_actions: bool,
    induce_rule_summary: bool,
    induce_scene_descriptions: bool,
    include_final_frames: bool,
) -> str:
    sections = [_core_prompt(output_mode)]
    if include_frames:
        sections.append(
            FRAME_CONTEXT_PROMPT
            if output_mode == "actions"
            else GRID_FRAME_CONTEXT_PROMPT
        )
    if include_final_frames:
        sections.append(
            FINAL_FRAME_CONTEXT_PROMPT
            if output_mode == "actions"
            else GRID_FINAL_FRAME_CONTEXT_PROMPT
        )
    if output_mode == "grid" or (include_textual_grids and output_mode != "actions"):
        sections.append(GRID_CONTEXT_PROMPT)
    if output_mode == "actions" or include_actions:
        sections.append(ACTION_CONTEXT_PROMPT)
    sections.append(
        _transformation_prompt(output_mode)
        + " "
        + _response_instruction(
            output_mode, induce_rule_summary, induce_scene_descriptions
        )
    )
    return "\n".join(sections)


def _core_prompt(output_mode: str) -> str:
    if output_mode == "actions":
        return EMBODIED_CORE_PROMPT
    if output_mode == "grid":
        return TEXTUAL_CORE_PROMPT
    raise ValueError(f"unsupported output mode: {output_mode}")


def _transformation_prompt(output_mode: str) -> str:
    if output_mode == "actions":
        return TRANSFORMATION_PROMPT
    if output_mode == "grid":
        return (
            "The demonstrations are examples, not an output to copy. Apply their "
            "shared rule to the test input and derive its new output grid."
        )
    raise ValueError(f"unsupported output mode: {output_mode}")


def _response_instruction(
    output_mode: str,
    induce_rule_summary: bool,
    induce_scene_descriptions: bool,
) -> str:
    if output_mode == "actions":
        result = 'Return only one JSON object with an "actions" list.'
    elif output_mode == "grid":
        result = "Return only one JSON object with the complete output_grid."
    else:
        raise ValueError(f"unsupported output mode: {output_mode}")
    descriptions = (
        ' Include "scene_descriptions" first, containing concise per-demonstration '
        "grid descriptions followed by a brief comparison."
        if induce_scene_descriptions
        else ""
    )
    if not induce_rule_summary:
        return result + descriptions
    if output_mode == "actions":
        return (
            """Return only one JSON
object with a concise textual "rule_summary" stating the shared transformation
and an "actions" list. The rule_summary is a conclusion, not a step-by-step
explanation."""
            + descriptions
            + "\n"
        )
    return (
        'Return only one JSON object with a concise textual "rule_summary" '
        "stating the shared transformation and the complete output_grid. The "
        "rule_summary is a conclusion, not a step-by-step explanation." + descriptions
    ) + "\n"


def _task_instruction(
    output_mode: str,
    induce_rule_summary: bool,
    include_frames: bool,
    include_actions: bool,
    include_grids: bool,
    include_final_frames: bool,
    induce_scene_descriptions: bool,
) -> str:
    """Give the concrete inference task again next to the held-out input."""
    test_input = "test input_grid" if include_grids else "test input"
    evidence: list[str] = []
    if include_grids:
        evidence.append("compare its input_grid with its output_grid")
    for enabled, context in (
        (include_frames, "use the table-camera video"),
        (include_final_frames, "use the final camera frames"),
        (include_actions, "use the action sequence"),
    ):
        if enabled:
            evidence.append(context)
    evidence_instruction = " and ".join(evidence)

    demonstration_answer = (
        "demonstration output_grid or action sequence"
        if include_grids and include_actions
        else "demonstration output_grid"
        if include_grids
        else "demonstration action sequence"
        if include_actions
        else "demonstration answer"
    )
    common = (
        f"For each demonstration, {evidence_instruction} to understand the "
        "transformation that produced that output. Infer one transformation "
        f"consistent with all demonstrations, then apply it to the {test_input}. "
        f"Do not copy any {demonstration_answer} as the test answer: the test "
        f"answer must be newly derived from the {test_input}. "
    )
    if induce_scene_descriptions:
        common += _scene_description_instruction(
            include_frames=include_frames,
            include_final_frames=include_final_frames,
        )
    rule_summary = (
        "with a concise rule_summary and " if induce_rule_summary else "with "
    )
    if output_mode == "actions":
        return (
            common + "Internally verify the resulting colored cells and coordinates, "
            "then construct the action plan for that new output. Return only the "
            "JSON object required by the supplied response schema, "
            + rule_summary
            + "the actions."
        )
    if output_mode == "grid":
        return (
            common + "Return only the JSON object required by the supplied response "
            "schema, " + rule_summary + "the complete newly derived output_grid."
        )
    raise ValueError(f"unsupported output mode: {output_mode}")


def _scene_description_instruction(
    *, include_frames: bool, include_final_frames: bool
) -> str:
    """Describe grid evidence while avoiding irrelevant video narration."""
    source = (
        "Focus exclusively on the input and output grid frames shown in each "
        "demonstration. "
        if include_frames or include_final_frames
        else "Focus on each demonstration's input_grid and output_grid. "
    )
    video_exclusion = (
        "Do not describe the robot, its motion, the camera, manipulation "
        "trajectories, or intermediate video events. "
        if include_frames
        else ""
    )
    return (
        "Before deriving the answer, fill scene_descriptions. "
        + source
        + video_exclusion
        + "For each demonstration, briefly describe the input grid, final output "
        "grid, and their visible differences using qualitative spatial language, "
        "not an exact grid reconstruction. Report only supported details. For each "
        "demonstration, describe its visible input/output evidence; reserve the "
        "shared-rule statement for rule_summary after the comparison. Then briefly "
        "compare the demonstrations by stating their shared input-to-output changes "
        "and differing details. Keep the descriptions concise, but include "
        "visually relevant objects, spatial relationships, and input-to-output "
        "changes needed to compare the demonstrations. "
    )


def _validate_prompt_inputs(
    task: VLMTaskInput,
    *,
    output_mode: str,
    include_frames: bool,
    include_textual_grids: bool,
    include_actions: bool,
    include_final_frames: bool,
    demonstration_video_urls: list[str] | None,
    demonstration_final_image_urls: list[list[tuple[str, str]]] | None,
    test_image_url: list[tuple[str, str]] | str | None,
) -> None:
    if output_mode == "actions" and not include_actions:
        raise ValueError("action output requires demonstration actions")
    if output_mode == "grid" and not include_textual_grids:
        raise ValueError("grid output requires textual demonstration grids")
    if not include_frames and not include_final_frames:
        if (
            demonstration_video_urls is not None
            or demonstration_final_image_urls is not None
            or test_image_url is not None
        ):
            raise ValueError("frame URLs must be omitted when frames are disabled")
        return
    if include_frames and (
        demonstration_video_urls is None
        or len(demonstration_video_urls) != len(task.demonstrations)
    ):
        raise ValueError(
            "one table-camera video data URL is required per demonstration"
        )
    if include_final_frames and (
        demonstration_final_image_urls is None
        or len(demonstration_final_image_urls) != len(task.demonstrations)
    ):
        raise ValueError("one final image data URL is required per selected camera")
    if not include_frames and demonstration_video_urls is not None:
        raise ValueError("video URLs require video context")
    if not include_final_frames and demonstration_final_image_urls is not None:
        raise ValueError("final image URLs require final-frame context")
    if test_image_url is None:
        raise ValueError("a test image data URL is required when frames are enabled")


def _normalize_camera_images(
    image: list[tuple[str, str]] | str | None,
) -> list[tuple[str, str]]:
    if image is None:
        return []
    return [("table_cam", image)] if isinstance(image, str) else image


def _demonstration_heading(
    index: int,
    demonstration: VLMDemonstration,
    *,
    include_frames: bool,
    include_final_frames: bool,
) -> str:
    dimensions = _format_demo_dimensions(demonstration)
    if not include_frames and not include_final_frames:
        return f"Demonstration {index}{dimensions}:"
    contexts: list[str] = []
    if include_frames:
        contexts.append(f"a video with {len(demonstration.rgb_frames)} sampled frames")
    if include_final_frames:
        contexts.append("final frame images from the selected cameras")
    if include_frames and not include_final_frames:
        return (
            f"Demonstration {index}{dimensions}: this video contains "
            f"{len(demonstration.rgb_frames)} sampled frames ordered from earliest to latest."
        )
    return f"Demonstration {index}{dimensions}: the following provide {' and '.join(contexts)}."


def _format_demo_dimensions(demonstration: VLMDemonstration) -> str:
    dimensions: list[str] = []
    if demonstration.input_grid_size is not None:
        rows, columns = demonstration.input_grid_size
        dimensions.append(f"input grid {rows}x{columns}")
    if demonstration.output_grid_size is not None:
        rows, columns = demonstration.output_grid_size
        dimensions.append(f"output grid {rows}x{columns}")
    return f" ({', '.join(dimensions)})" if dimensions else ""


def _test_heading(
    input_grid_size: tuple[int, int] | None, *, include_visual_context: bool
) -> str:
    dimensions = (
        f" Its input grid is {input_grid_size[0]}x{input_grid_size[1]}."
        if input_grid_size is not None
        else ""
    )
    frame_context = " this is only its initial frame." if include_visual_context else ""
    return (
        f"Test input:{frame_context}{dimensions} Infer the shared transformation "
        "and the output grid size."
    )


def _image_part(url: str | None) -> dict[str, Any]:
    if url is None:
        raise ValueError("image URL is required")
    return {"type": "image_url", "image_url": {"url": url}}


def _video_part(url: str) -> dict[str, Any]:
    # vLLM's OpenAI wire format uses video_url; the data:video MIME type makes
    # Qwen process the comma-separated JPEG frames as one video modality.
    return {"type": "video_url", "video_url": {"url": url}}
