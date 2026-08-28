import re

import pytest
import src.vlm_prompts as prompts
import torch
from src.datasets.actions import HighLevelActionEvent
from src.vlm_preprocessing import VLMDemonstration, VLMTaskInput
from src.vlm_prompts import (
    ACTION_CONTEXT_PROMPT,
    GRID_CONTEXT_PROMPT,
    GRID_FRAME_CONTEXT_PROMPT,
    RENDERED_SCENE_BACKGROUND_PROMPT,
    build_vlm_messages,
    format_demonstration_actions,
    response_schema_for_output_mode,
)


def _task() -> VLMTaskInput:
    return VLMTaskInput(
        task_id="secret_task_id",
        test_key="test_0",
        demonstrations=[
            VLMDemonstration(
                rgb_frames=torch.zeros((2, 4, 4, 3), dtype=torch.uint8),
                actions=[
                    HighLevelActionEvent((1, 2, 0, -1, -1), 0, 2),
                    HighLevelActionEvent((2, 2, 1, 0, 1), 3, 6),
                ],
                input_grid=[[2, 0, 3], [4, 5, 6]],
                output_grid=[[2, 3]],
                input_grid_size=(2, 3),
                output_grid_size=(1, 2),
            )
        ],
        test_first_frame=torch.zeros((4, 4, 3), dtype=torch.uint8),
        test_input_grid=[[8, 0], [0, 1]],
        test_input_grid_size=(2, 2),
    )


@pytest.mark.parametrize(
    ("output_mode", "induce_rule_summary", "expected_required"),
    [
        ("actions", True, {"actions", "rule_summary"}),
        ("actions", False, {"actions"}),
        ("grid", True, {"output_grid", "rule_summary"}),
        ("grid", False, {"output_grid"}),
    ],
)
def test_response_schema_matches_rule_summary_setting(
    output_mode: str, induce_rule_summary: bool, expected_required: set[str]
) -> None:
    schema = response_schema_for_output_mode(output_mode, induce_rule_summary)

    assert set(schema["required"]) == expected_required
    assert ("rule_summary" in schema["properties"]) is induce_rule_summary
    if induce_rule_summary:
        assert next(iter(schema["properties"])) == "rule_summary"


EXPECTED_EMBODIED_CORE = (
    "You solve embodied ARC tasks from solved robot demonstrations.\n"
)

EXPECTED_GRID_CORE = """You solve ARC tasks from solved grid demonstrations.

Infer the single abstract transformation shared by the demonstrations, apply it
to the test input, and return the complete newly derived output grid.
"""

EXPECTED_ACTION_CONTEXT = """Every action is a JSON object with the categorical fields phase, color_id,
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

EXPECTED_GRID_CONTEXT = """Each textual demonstration contains an input_grid and
the corresponding output_grid. Compare every pair to infer one transformation,
then apply it to the test input_grid. Each grid is rectangular. Grid values are
ARC color IDs: 0 black/background, 1 blue, 2 red, 3 green, 4 yellow, 5 gray,
6 magenta, 7 orange, 8 teal, and 9 brown. Preserve the complete rectangular
output grid, including black/background cells.
"""

EXPECTED_FRAME_CONTEXT = f"""Each demonstration is an ordered sequence of sampled `table_cam` RGB frames. The scene shows
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

EXPECTED_GRID_FRAME_CONTEXT = GRID_FRAME_CONTEXT_PROMPT

EXPECTED_TRANSFORMATION = """The demonstrations are examples, not a plan to copy. Apply their shared rule to
the test input and generate the actions for its new output."""
EXPECTED_GRID_TRANSFORMATION = (
    "The demonstrations are examples, not an output to copy. Apply their shared "
    "rule to the test input and derive its new output grid."
)

EXPECTED_ACTION_RESPONSE = """Return only one JSON
object with a concise textual "rule_summary" stating the shared transformation
and an "actions" list. The rule_summary is a conclusion, not a step-by-step
explanation.
"""
EXPECTED_GRID_RESPONSE = (
    'Return only one JSON object with a concise textual "rule_summary" stating '
    "the shared transformation and the complete output_grid. The rule_summary "
    "is a conclusion, not a step-by-step explanation.\n"
)
EXPECTED_ACTION_FRAMES_ACTIONS_TASK = (
    "For each demonstration, use the table-camera video and use the action sequence to understand the "
    "transformation that produced that output. Infer one "
    "transformation consistent with all demonstrations, then apply it to the test "
    "input. Do not copy any demonstration action sequence as the test answer: the "
    "test answer must be newly derived from the test input. Internally "
    "verify the resulting colored cells and coordinates, then construct the action "
    "plan for that new output. Return only the JSON object required by the supplied "
    "response schema, with a concise rule_summary and the actions."
)
EXPECTED_ACTION_ALL_CONTEXT_TASK = (
    "For each demonstration, compare its input_grid with its output_grid and use "
    "the table-camera video and use the action sequence to understand the transformation that produced "
    "that output. Infer one transformation consistent with all "
    "demonstrations, then apply it to the test input_grid. Do not copy any "
    "demonstration output_grid or action sequence as the test answer: the test "
    "answer must be newly derived from the test input_grid. Internally verify the resulting colored "
    "cells and coordinates, then construct the action plan for that new output. "
    "Return only the JSON object required by the supplied response schema, with a "
    "concise rule_summary and the actions."
)
EXPECTED_ACTION_GRIDS_ACTIONS_TASK = (
    "For each demonstration, compare its input_grid with its output_grid and use "
    "the action sequence to understand the transformation that produced that output. "
    "Infer one transformation consistent with all demonstrations, then apply it "
    "to the test input_grid. Do not copy any demonstration output_grid or action "
    "sequence as the test answer: the test answer must be newly derived from the "
    "test input_grid. Internally verify the resulting colored cells and coordinates, "
    "then construct the action plan for that new output. Return only the JSON "
    "object required by the supplied response schema, with a concise rule_summary "
    "and the actions."
)
EXPECTED_GRID_ONLY_TASK = (
    "For each demonstration, compare its input_grid with its output_grid to "
    "understand the transformation that produced that output. Infer one "
    "transformation consistent with all demonstrations, then apply it to the test "
    "input_grid. Do not copy any demonstration output_grid as the test answer: the "
    "test answer must be newly derived from the test input_grid. "
    "Return only the JSON object required by the supplied response schema, with a "
    "concise rule_summary and the complete newly derived output_grid."
)
EXPECTED_GRID_ACTIONS_TASK = (
    "For each demonstration, compare its input_grid with its output_grid and use "
    "the action sequence to understand the transformation that produced that output. "
    "Infer one transformation consistent with all demonstrations, then apply it "
    "to the test input_grid. Do not copy any demonstration output_grid or action "
    "sequence as the test answer: the test answer must be newly derived from the "
    "test input_grid. Return only the JSON object required by the supplied response "
    "schema, with a concise rule_summary and the complete newly derived "
    "output_grid."
)
EXPECTED_GRID_FRAMES_TASK = (
    "For each demonstration, compare its input_grid with its output_grid and use "
    "the table-camera video to understand the transformation that produced that output. "
    "Infer one transformation consistent with all demonstrations, then apply it "
    "to the test input_grid. Do not copy any demonstration output_grid as the test "
    "answer: the test answer must be newly derived from the test input_grid. Return "
    "only the JSON object required by the supplied response "
    "schema, with a concise rule_summary and the complete newly derived "
    "output_grid."
)
EXPECTED_GRID_ALL_CONTEXT_TASK = (
    "For each demonstration, compare its input_grid with its output_grid and use "
    "the table-camera video and use the action sequence to understand the transformation that produced "
    "that output. Infer one transformation consistent with all "
    "demonstrations, then apply it to the test input_grid. Do not copy any "
    "demonstration output_grid or action sequence as the test answer: the test "
    "answer must be newly derived from the test input_grid. "
    "Return only the JSON object required by "
    "the supplied response schema, with a concise rule_summary and the complete "
    "newly derived output_grid."
)
EXPECTED_DEMO_FRAMES_HEADING = (
    "Demonstration 1 (input grid 2x3, output grid 1x2): this video contains 2 "
    "sampled frames ordered from earliest to latest."
)
EXPECTED_DEMO_TEXT_HEADING = "Demonstration 1 (input grid 2x3, output grid 1x2):"
EXPECTED_TEST_FRAME_HEADING = (
    "Test input: this is only its initial frame. Its input grid is 2x2. Infer "
    "the shared transformation and the output grid size."
)
EXPECTED_TEST_TEXT_HEADING = (
    "Test input: Its input grid is 2x2. Infer the shared transformation and "
    "the output grid size."
)

DEMO_VIDEO = "data:video/jpeg;base64,demo"
TEST_IMAGE = "data:image/png;base64,test"


def _expected_content(*parts: str) -> list[dict]:
    return [
        {
            "type": "video_url",
            "video_url": {"url": DEMO_VIDEO},
        }
        if part == "demo_frames"
        else {
            "type": "image_url",
            "image_url": {"url": TEST_IMAGE},
        }
        if part == "test_frame"
        else {"type": "text", "text": part}
        for part in parts
    ]


@pytest.mark.parametrize(
    (
        "output_mode, include_frames, include_textual_grids, include_actions, "
        "expected_system_prompt, expected_content"
    ),
    [
        pytest.param(
            "actions",
            True,
            False,
            True,
            "\n".join(
                [
                    EXPECTED_EMBODIED_CORE,
                    EXPECTED_FRAME_CONTEXT,
                    EXPECTED_ACTION_CONTEXT,
                    EXPECTED_TRANSFORMATION + " " + EXPECTED_ACTION_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_FRAMES_HEADING,
                "demo_frames",
                "<demo_actions>",
                EXPECTED_TEST_FRAME_HEADING,
                "test_frame",
                EXPECTED_ACTION_FRAMES_ACTIONS_TASK,
            ),
            id="embodied-frames-actions",
        ),
        pytest.param(
            "actions",
            True,
            True,
            True,
            "\n".join(
                [
                    EXPECTED_EMBODIED_CORE,
                    EXPECTED_FRAME_CONTEXT,
                    EXPECTED_ACTION_CONTEXT,
                    EXPECTED_TRANSFORMATION + " " + EXPECTED_ACTION_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_FRAMES_HEADING,
                "demo_frames",
                "<demo_grids>",
                "<demo_actions>",
                EXPECTED_TEST_FRAME_HEADING,
                "test_frame",
                "<test_grid>",
                EXPECTED_ACTION_ALL_CONTEXT_TASK,
            ),
            id="embodied-frames-grids-actions",
        ),
        pytest.param(
            "actions",
            False,
            True,
            True,
            "\n".join(
                [
                    EXPECTED_EMBODIED_CORE,
                    EXPECTED_ACTION_CONTEXT,
                    EXPECTED_TRANSFORMATION + " " + EXPECTED_ACTION_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_TEXT_HEADING,
                "<demo_grids>",
                "<demo_actions>",
                EXPECTED_TEST_TEXT_HEADING,
                "<test_grid>",
                EXPECTED_ACTION_GRIDS_ACTIONS_TASK,
            ),
            id="embodied-grids-actions",
        ),
        pytest.param(
            "grid",
            False,
            True,
            False,
            "\n".join(
                [
                    EXPECTED_GRID_CORE,
                    EXPECTED_GRID_CONTEXT,
                    EXPECTED_GRID_TRANSFORMATION + " " + EXPECTED_GRID_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_TEXT_HEADING,
                "<demo_grids>",
                EXPECTED_TEST_TEXT_HEADING,
                "<test_grid>",
                EXPECTED_GRID_ONLY_TASK,
            ),
            id="grid-focused-grids",
        ),
        pytest.param(
            "grid",
            False,
            True,
            True,
            "\n".join(
                [
                    EXPECTED_GRID_CORE,
                    EXPECTED_GRID_CONTEXT,
                    EXPECTED_ACTION_CONTEXT,
                    EXPECTED_GRID_TRANSFORMATION + " " + EXPECTED_GRID_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_TEXT_HEADING,
                "<demo_grids>",
                "<demo_actions>",
                EXPECTED_TEST_TEXT_HEADING,
                "<test_grid>",
                EXPECTED_GRID_ACTIONS_TASK,
            ),
            id="grid-focused-grids-actions",
        ),
        pytest.param(
            "grid",
            True,
            True,
            False,
            "\n".join(
                [
                    EXPECTED_GRID_CORE,
                    EXPECTED_GRID_FRAME_CONTEXT,
                    EXPECTED_GRID_CONTEXT,
                    EXPECTED_GRID_TRANSFORMATION + " " + EXPECTED_GRID_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_FRAMES_HEADING,
                "demo_frames",
                "<demo_grids>",
                EXPECTED_TEST_FRAME_HEADING,
                "test_frame",
                "<test_grid>",
                EXPECTED_GRID_FRAMES_TASK,
            ),
            id="grid-focused-frames-grids",
        ),
        pytest.param(
            "grid",
            True,
            True,
            True,
            "\n".join(
                [
                    EXPECTED_GRID_CORE,
                    EXPECTED_GRID_FRAME_CONTEXT,
                    EXPECTED_GRID_CONTEXT,
                    EXPECTED_ACTION_CONTEXT,
                    EXPECTED_GRID_TRANSFORMATION + " " + EXPECTED_GRID_RESPONSE,
                ]
            ),
            _expected_content(
                EXPECTED_DEMO_FRAMES_HEADING,
                "demo_frames",
                "<demo_grids>",
                "<demo_actions>",
                EXPECTED_TEST_FRAME_HEADING,
                "test_frame",
                "<test_grid>",
                EXPECTED_GRID_ALL_CONTEXT_TASK,
            ),
            id="grid-focused-frames-grids-actions",
        ),
    ],
)
def test_final_prompt_composition(
    output_mode: str,
    include_frames: bool,
    include_textual_grids: bool,
    include_actions: bool,
    expected_system_prompt: str,
    expected_content: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specify the complete final prompt independently for every valid case."""
    monkeypatch.setattr(
        prompts, "format_demonstration_grids", lambda _demonstration: "<demo_grids>"
    )
    monkeypatch.setattr(
        prompts, "format_demonstration_actions", lambda _actions: "<demo_actions>"
    )
    monkeypatch.setattr(prompts, "format_test_input_grid", lambda _grid: "<test_grid>")
    messages = build_vlm_messages(
        _task(),
        output_mode=output_mode,
        include_frames=include_frames,
        include_textual_grids=include_textual_grids,
        include_actions=include_actions,
        induce_rule_summary=True,
        demonstration_video_urls=[DEMO_VIDEO] if include_frames else None,
        test_image_url=TEST_IMAGE if include_frames else None,
    )

    assert messages == [
        {"role": "system", "content": expected_system_prompt},
        {"role": "user", "content": expected_content},
    ]


def test_builds_embodied_prompt_from_frames_actions_and_optional_grids() -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="actions",
        include_frames=True,
        include_textual_grids=True,
        include_actions=True,
        induce_rule_summary=True,
        demonstration_video_urls=["data:video/jpeg;base64,frame-0,frame-1"],
        test_image_url="data:test",
    )
    system_prompt = messages[0]["content"]
    content = messages[1]["content"]
    serialized = str(messages)

    assert "secret_task_id" not in serialized
    assert [part["type"] for part in content].count("video_url") == 1
    assert [part["type"] for part in content].count("image_url") == 1
    assert "input_grid: [[2, 0, 3], [4, 5, 6]]" in serialized
    assert "output_grid: [[2, 3]]" in serialized
    assert any(
        part.get("text") == "Test ARC input grid:\ninput_grid: [[8, 0], [0, 1]]"
        for part in content
    )
    assert '"phase": "grasp"' in serialized
    assert "grasp a red cube from the reserve" in serialized
    assert ACTION_CONTEXT_PROMPT in system_prompt
    assert GRID_CONTEXT_PROMPT not in system_prompt
    assert "scene shows" in system_prompt.lower()
    assert "an arc input grid on the left" in system_prompt.lower()
    assert 'concise textual "rule_summary"' in system_prompt
    assert "Infer the shared transformation and the output grid size" in serialized
    assert "compare its input_grid with its output_grid" in content[-1]["text"]


def test_builds_grid_focused_prompt_with_optional_frames_and_actions() -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=False,
        include_textual_grids=True,
        include_actions=True,
        induce_rule_summary=False,
    )
    system_prompt = messages[0]["content"]
    content = messages[1]["content"]
    serialized = str(messages)

    assert all(part["type"] == "text" for part in content)
    assert "input_grid: [[2, 0, 3], [4, 5, 6]]" in serialized
    assert '"phase": "grasp"' in serialized
    assert GRID_CONTEXT_PROMPT in system_prompt
    assert ACTION_CONTEXT_PROMPT in system_prompt
    assert "scene shows an ARC input grid on the left" not in system_prompt
    assert "rule_summary" not in serialized


def test_builds_grid_focused_prompt_from_final_images() -> None:
    final_image = "data:image/png;base64,final"
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=False,
        include_textual_grids=True,
        include_actions=False,
        induce_rule_summary=True,
        include_final_frames=True,
        demonstration_final_image_urls=[[("table_cam", final_image)]],
        test_image_url=TEST_IMAGE,
    )
    system_prompt = messages[0]["content"]
    content = messages[1]["content"]

    assert "one final RGB image" in system_prompt
    assert "ordered sequence of sampled RGB frames" not in system_prompt
    assert [part["type"] for part in content].count("video_url") == 0
    assert [part["type"] for part in content].count("image_url") == 2
    assert [part["text"] for part in content if part["type"] == "text"][:2] == [
        (
            "Demonstration 1 (input grid 2x3, output grid 1x2): the following provide "
            "final frame images from the selected cameras."
        ),
        "Final frame — camera perspective: table_cam",
    ]
    assert content[2]["image_url"]["url"] == final_image
    assert "use the final camera frames" in content[-1]["text"]


def test_scene_descriptions_focus_on_grid_frames_without_video_language() -> None:
    final_image = "data:image/png;base64,final"
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=False,
        include_textual_grids=True,
        include_actions=False,
        induce_rule_summary=True,
        induce_scene_descriptions=True,
        include_final_frames=True,
        demonstration_final_image_urls=[[("table_cam", final_image)]],
        test_image_url=TEST_IMAGE,
    )
    instruction = messages[1]["content"][-1]["text"]
    schema = response_schema_for_output_mode(
        "grid", induce_rule_summary=True, induce_scene_descriptions=True
    )

    assert "Focus exclusively on the input and output grid frames" in instruction
    assert "robot, its motion" not in instruction
    assert "intermediate video events" not in instruction
    assert "reserve the shared-rule statement for rule_summary" in instruction
    assert "most 2–3 sentences" not in instruction
    assert list(schema["properties"]) == [
        "scene_descriptions",
        "rule_summary",
        "output_grid",
    ]
    assert set(schema["required"]) == {
        "scene_descriptions",
        "rule_summary",
        "output_grid",
    }


def test_scene_descriptions_exclude_video_narration_when_videos_are_present() -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=True,
        include_textual_grids=True,
        include_actions=False,
        induce_rule_summary=True,
        induce_scene_descriptions=True,
        demonstration_video_urls=[DEMO_VIDEO],
        test_image_url=TEST_IMAGE,
    )

    instruction = messages[1]["content"][-1]["text"]
    assert "Do not describe the robot, its motion" in instruction
    assert "intermediate video events" in instruction


def test_allows_video_and_final_image_prompt() -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=True,
        include_final_frames=True,
        include_textual_grids=True,
        include_actions=False,
        induce_rule_summary=True,
        demonstration_video_urls=[DEMO_VIDEO],
        demonstration_final_image_urls=[[("table_cam", "data:final")]],
        test_image_url=TEST_IMAGE,
    )
    assert sum(part["type"] == "video_url" for part in messages[1]["content"]) == 1
    assert sum(part["type"] == "image_url" for part in messages[1]["content"]) == 2


def test_disabled_context_is_absent_from_prompt() -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=False,
        include_textual_grids=True,
        include_actions=False,
        induce_rule_summary=False,
    )
    serialized = str(messages)

    assert "video_url" not in serialized
    assert "image_url" not in serialized
    assert "Compressed high-level actions" not in serialized
    assert ACTION_CONTEXT_PROMPT not in messages[0]["content"]
    assert GRID_CONTEXT_PROMPT in messages[0]["content"]
    assert "input_grid: [[2, 0, 3], [4, 5, 6]]" in serialized
    lowered = serialized.lower()
    for absent_term in (
        r"actions?",
        r"frames?",
        r"videos?",
        r"images?",
        r"robots?",
        r"reserves?",
        r"cubes?",
        r"grasps?",
        r"places?",
    ):
        assert re.search(rf"\b{absent_term}\b", lowered) is None


@pytest.mark.parametrize(
    "include_frames, include_actions, expected_terms, absent_terms",
    [
        pytest.param(
            False,
            False,
            ("compare its input_grid with its output_grid",),
            ("ordered frames", "action sequence", "robot frames and actions"),
            id="grids-only",
        ),
        pytest.param(
            True,
            False,
            ("compare its input_grid with its output_grid", "table-camera video"),
            ("action sequence",),
            id="grids-frames",
        ),
        pytest.param(
            False,
            True,
            ("compare its input_grid with its output_grid", "action sequence"),
            ("ordered frames", "robot frames and actions"),
            id="grids-actions",
        ),
        pytest.param(
            True,
            True,
            (
                "compare its input_grid with its output_grid",
                "table-camera video",
                "action sequence",
            ),
            (),
            id="all-contexts",
        ),
    ],
)
def test_task_instruction_names_exactly_the_enabled_contexts(
    include_frames: bool,
    include_actions: bool,
    expected_terms: tuple[str, ...],
    absent_terms: tuple[str, ...],
) -> None:
    messages = build_vlm_messages(
        _task(),
        output_mode="grid",
        include_frames=include_frames,
        include_textual_grids=True,
        include_actions=include_actions,
        induce_rule_summary=True,
        demonstration_video_urls=[DEMO_VIDEO] if include_frames else None,
        test_image_url=TEST_IMAGE if include_frames else None,
    )
    instruction = messages[1]["content"][-1]["text"]

    for expected_term in expected_terms:
        assert expected_term in instruction
    for absent_term in absent_terms:
        assert absent_term not in instruction


def test_rejects_frame_urls_when_frames_are_disabled() -> None:
    try:
        build_vlm_messages(
            _task(),
            output_mode="grid",
            include_frames=False,
            include_textual_grids=True,
            include_actions=False,
            induce_rule_summary=True,
            demonstration_video_urls=[],
        )
    except ValueError as error:
        assert str(error) == "frame URLs must be omitted when frames are disabled"
    else:
        raise AssertionError("disabled frames should reject media URLs")


def test_rejects_an_output_without_its_required_demonstration_context() -> None:
    try:
        build_vlm_messages(
            _task(),
            output_mode="actions",
            include_frames=True,
            include_textual_grids=False,
            include_actions=False,
            induce_rule_summary=True,
            demonstration_video_urls=["data:video/jpeg;base64,frame-0,frame-1"],
            test_image_url="data:test",
        )
    except ValueError as error:
        assert str(error) == "action output requires demonstration actions"
    else:
        raise AssertionError("action output requires demonstration actions")


def test_formats_empty_demonstration_actions() -> None:
    assert format_demonstration_actions([]) == (
        "Compressed high-level actions: [] (no colored output cubes)."
    )
