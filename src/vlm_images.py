"""Convert sampled RGB tensors into image data URLs for multimodal requests."""

from __future__ import annotations

import base64
from io import BytesIO
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


def frames_to_mp4_data_url(frames: torch.Tensor, *, fps: float) -> str:
    """Encode a complete RGB frame sequence as one base64 MP4 video."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("RGB frames must have shape [T, H, W, 3]")
    if len(frames) == 0:
        raise ValueError("a video must contain at least one frame")
    if frames.dtype != torch.uint8:
        raise TypeError("RGB frames must use torch.uint8")
    if fps <= 0:
        raise ValueError("video fps must be positive")

    try:
        import imageio.v2 as imageio
    except ImportError as error:  # pragma: no cover - dependency failure on host
        raise RuntimeError("full-video diagnostics require imageio[ffmpeg]") from error

    with TemporaryDirectory(prefix="era-bench-video-") as temporary_directory:
        video_path = f"{temporary_directory}/episode.mp4"
        writer = imageio.get_writer(
            video_path,
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
        with open(video_path, "rb") as video_file:
            encoded = base64.b64encode(video_file.read()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"
