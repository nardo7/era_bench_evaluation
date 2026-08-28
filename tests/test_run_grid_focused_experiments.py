from scripts.run_experiments import (
    EMBODIED_EXPERIMENTS,
    GRID_FOCUSED_EXPERIMENTS,
    experiment_configs,
)
from src.evaluations.evaluation_config import (
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    GenerationConfig,
    ModelConfig,
    ServerConfig,
)


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        dataset=DatasetConfig(paths=[]),
        model=ModelConfig(model_id="model", served_name="model"),
        server=ServerConfig(),
        generation=GenerationConfig(),
        evaluation=EvaluationRunConfig(output_dir="output"),
    )


def test_yields_default_embodied_experiment_mask() -> None:
    experiments = list(experiment_configs(_config()))

    assert len(experiments) == len(EMBODIED_EXPERIMENTS)
    assert {experiment.evaluation.prompt_mode for experiment in experiments} == {
        "embodied"
    }
    assert all(experiment.evaluation.include_actions for experiment in experiments)
    assert all(
        experiment.evaluation.cameras == ("table_cam",) for experiment in experiments
    )
    assert (
        sum(experiment.evaluation.include_final_frames for experiment in experiments)
        == sum(experiment["include_final_frames"] for experiment in EMBODIED_EXPERIMENTS)
    )


def test_yields_grid_focused_experiment_mask() -> None:
    experiments = list(experiment_configs(_config(), "grid_focused"))

    assert len(experiments) == len(GRID_FOCUSED_EXPERIMENTS)
    assert {experiment.evaluation.prompt_mode for experiment in experiments} == {
        "grid_focused"
    }
    assert all(
        experiment.evaluation.include_textual_grids for experiment in experiments
    )
