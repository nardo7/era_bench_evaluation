import base64
from io import BytesIO

import torch
from PIL import Image

from src.vlm_images import frame_to_data_url, frames_to_video_data_url


def test_encodes_ordered_jpeg_frames_as_one_video_data_url() -> None:
    frames = torch.zeros((2, 3, 4, 3), dtype=torch.uint8)
    frames[1, :, :, 0] = 255

    url = frames_to_video_data_url(frames)
    encoded_frames = url.removeprefix("data:video/jpeg;base64,").split(",")

    assert url.startswith("data:video/jpeg;base64,")
    assert len(encoded_frames) == 2
    image = Image.open(BytesIO(base64.b64decode(encoded_frames[1])))
    assert image.size == (4, 3)
    red, green, blue = image.getpixel((0, 0))
    assert red > 240 and green < 20 and blue < 20


def test_encodes_test_frame_as_png_image_url() -> None:
    frame = torch.zeros((3, 4, 3), dtype=torch.uint8)
    assert frame_to_data_url(frame).startswith("data:image/png;base64,")
