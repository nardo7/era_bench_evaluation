"""Convert sampled RGB tensors into image data URLs for multimodal requests."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image


def upscale_rgb_frames(frames: torch.Tensor, *, factor: int) -> torch.Tensor:
    """Nearest-neighbor upscale RGB frame tensors without cropping the scene."""
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("image scale factor must be a positive integer")
    if frames.ndim not in (3, 4) or frames.shape[-1] != 3:
        raise ValueError("RGB frames must have shape [H, W, 3] or [T, H, W, 3]")
    if frames.dtype != torch.uint8:
        raise TypeError("RGB frames must use torch.uint8")
    if factor == 1:
        return frames
    return frames.repeat_interleave(factor, dim=-3).repeat_interleave(factor, dim=-2)


def frame_to_data_url(frame: torch.Tensor) -> str:
    """Encode one ``[H, W, 3]`` uint8 RGB frame as an in-memory PNG URL."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("an RGB frame must have shape [H, W, 3]")
    if frame.dtype != torch.uint8:
        raise TypeError("RGB frames must use torch.uint8")

    array = frame.detach().cpu().contiguous().numpy()
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def frames_to_video_data_url(frames: torch.Tensor) -> str:
    """Encode ordered JPEG frames as Qwen/vLLM's in-memory video format."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("RGB frames must have shape [T, H, W, 3]")
    if len(frames) == 0:
        raise ValueError("a video must contain at least one frame")
    encoded_frames: list[str] = []
    for frame in frames:
        if frame.dtype != torch.uint8:
            raise TypeError("RGB frames must use torch.uint8")
        array = frame.detach().cpu().contiguous().numpy()
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG")
        encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return f"data:video/jpeg;base64,{','.join(encoded_frames)}"


def demonstration_video_data_urls(
    *,
    task_id: str,
    frames: Sequence[torch.Tensor],
    video_format: str,
    video_cache_dir: Path,
    video_cache_namespace: str,
    frame_stride: int,
    fps: float,
) -> list[str]:
    """Encode or load table-camera demonstration videos for one task."""
    if video_format == "jpeg_frames":
        return [frames_to_video_data_url(demonstration_frames) for demonstration_frames in frames]

    return [
        _cached_mp4_data_url(
            demonstration_frames,
            cache_path=_mp4_cache_path(
                task_id=task_id,
                demo_index=demo_index,
                video_cache_dir=video_cache_dir,
                video_cache_namespace=video_cache_namespace,
                frame_stride=frame_stride,
                fps=fps,
            ),
            fps=fps,
        )
        for demo_index, demonstration_frames in enumerate(frames)
    ]


def video_cache_namespace(dataset_paths: Sequence[Path]) -> str:
    """Identify source datasets without reading their frame payloads."""
    return json.dumps(
        [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in dataset_paths
        ],
        sort_keys=True,
    )


def _cached_mp4_data_url(frames: torch.Tensor, *, cache_path: Path, fps: float) -> str:
    if not cache_path.is_file():
        frames_to_mp4_file(frames, fps=fps, output_path=cache_path)
    return mp4_file_to_data_url(cache_path)


def _mp4_cache_path(
    *,
    task_id: str,
    demo_index: int,
    video_cache_dir: Path,
    video_cache_namespace: str,
    frame_stride: int,
    fps: float,
) -> Path:
    cache_key = json.dumps(
        {
            "version": 1,
            "dataset": video_cache_namespace,
            "task_id": task_id,
            "demo_index": demo_index,
            "frame_stride": frame_stride,
            "fps": fps,
        },
        sort_keys=True,
    ).encode("utf-8")
    return video_cache_dir / f"{hashlib.sha256(cache_key).hexdigest()}.mp4"


def frames_to_mp4_file(
    frames: torch.Tensor, *, fps: float, output_path: Path
) -> None:
    """Encode a complete RGB frame sequence as an MP4 file."""
    _validate_video_frames(frames, fps)

    try:
        import imageio.v2 as imageio
    except ImportError as error:  # pragma: no cover - dependency failure on host
        raise RuntimeError("MP4 video encoding requires imageio[ffmpeg]") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="era-bench-video-", dir=output_path.parent
    ) as directory:
        temporary_path = Path(directory) / output_path.name
        writer = imageio.get_writer(
            temporary_path,
            format="mp4",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        try:
            for frame in frames:
                writer.append_data(frame.detach().cpu().contiguous().numpy())
        finally:
            writer.close()
        temporary_path.replace(output_path)


def mp4_file_to_data_url(video_path: Path) -> str:
    """Read an MP4 file as a base64 data URL for a chat-completions request."""
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def frames_to_mp4_data_url(frames: torch.Tensor, *, fps: float) -> str:
    """Encode a complete RGB frame sequence as one base64 MP4 video."""
    _validate_video_frames(frames, fps)

    with TemporaryDirectory(prefix="era-bench-video-") as directory:
        video_path = Path(directory) / "episode.mp4"
        frames_to_mp4_file(frames, fps=fps, output_path=video_path)
        return mp4_file_to_data_url(video_path)


def _validate_video_frames(frames: torch.Tensor, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("RGB frames must have shape [T, H, W, 3]")
    if len(frames) == 0:
        raise ValueError("a video must contain at least one frame")
    if frames.dtype != torch.uint8:
        raise TypeError("RGB frames must use torch.uint8")
    if fps <= 0:
        raise ValueError("video fps must be positive")
