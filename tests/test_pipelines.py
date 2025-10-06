from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV erforderlich", exc_type=ImportError)
import numpy as np
from PIL import Image

from src.image_pipeline import process_image
from src.utils import ProcessingOptions
from src.video_pipeline import process_video


def _base_options(tmp_path: Path) -> ProcessingOptions:
    return ProcessingOptions(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        face_detection_enabled=False,
    )


def test_process_image(tmp_path: Path) -> None:
    src = tmp_path / "img.jpg"
    arr = np.zeros((600, 800, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src)
    options = _base_options(tmp_path)
    result = process_image(src, options, None)
    assert result.target.exists()
    img = Image.open(result.target)
    assert img.size == (options.size, options.size)


def test_process_video(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    width, height = 640, 360
    fps = 5
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(5):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 40
        writer.write(frame)
    writer.release()

    options = _base_options(tmp_path)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    result = process_video(src, options, None)
    assert result.target.exists()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(result.target),
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    width_out, height_out = map(int, probe.stdout.decode("utf-8").strip().split(","))
    assert width_out == options.size
    assert height_out == options.size
