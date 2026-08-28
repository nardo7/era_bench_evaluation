import importlib.util
from pathlib import Path

import h5py
import numpy as np

_CREATE_VIDEOS_PATH = Path(__file__).parents[1] / "scripts" / "create_videos.py"
_module_spec = importlib.util.spec_from_file_location(
    "create_videos", _CREATE_VIDEOS_PATH
)
assert _module_spec is not None and _module_spec.loader is not None
create_videos = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(create_videos)


def test_downsamples_source_frames_before_writing(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "era_bench.hdf5"
    source_frames = np.zeros((75, 2, 2, 3), dtype=np.uint8)
    source_frames[:, :, :, 0] = np.arange(75, dtype=np.uint8)[:, None, None]
    with h5py.File(dataset_path, "w") as dataset:
        dataset.create_group("data").create_group("demo_0").create_group(
            "obs"
        ).create_dataset("table_cam", data=source_frames)

    captured: dict[str, object] = {}

    def capture_video(frames: np.ndarray, output_path, fps: int) -> None:
        captured["frames"] = frames
        captured["output_path"] = output_path
        captured["fps"] = fps

    monkeypatch.setattr(create_videos, "_write_video", capture_video)

    assert create_videos.create_videos(
        dataset_path, tmp_path / "videos", fps=1, source_fps=25
    ) == (1, 0)
    assert captured["frames"][:, 0, 0, 0].tolist() == [0, 37, 74]
    assert captured["fps"] == 1


def test_rejects_output_fps_above_source_fps(tmp_path) -> None:
    with np.testing.assert_raises_regex(
        ValueError, r"output FPS \(30\) cannot exceed source FPS \(25\)"
    ):
        create_videos.create_videos(
            tmp_path / "missing.hdf5", tmp_path / "videos", fps=30, source_fps=25
        )
