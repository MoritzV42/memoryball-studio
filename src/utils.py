"""Utility functions for memoryball-autocrop."""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass(slots=True)
class CropBox:
    """Represents a square crop box."""

    x: float
    y: float
    size: float

    def as_tuple(self) -> Tuple[int, int, int, int]:
        left = math.floor(self.x)
        top = math.floor(self.y)
        right = math.ceil(self.x + self.size)
        bottom = math.ceil(self.y + self.size)
        return left, top, right, bottom


CROP_OVERFLOW_RATIO = 0.5
ORIENTATION_CIRCLE_MARGIN = 0.1


def _circle_fraction(circle_margin: float) -> float:
    """Return the fraction of a square that remains after applying the circle margin."""

    if circle_margin <= 0.0:
        return 1.0
    return max(0.0, 1.0 - 2.0 * circle_margin)


def square_size_for_circle(diameter: float, circle_margin: float = ORIENTATION_CIRCLE_MARGIN) -> float:
    """Return the side length of a square whose inscribed circle matches ``diameter``."""

    diameter = max(0.0, float(diameter))
    fraction = _circle_fraction(circle_margin)
    if fraction <= 0.0:
        return diameter
    return diameter / fraction if diameter else 0.0


def expand_crop_for_circle(crop: CropBox, circle_margin: float = ORIENTATION_CIRCLE_MARGIN) -> CropBox:
    """Expand ``crop`` so that the orientation circle matches the detected region."""

    fraction = _circle_fraction(circle_margin)
    if fraction <= 0.0:
        return CropBox(crop.x, crop.y, crop.size)
    new_size = crop.size / fraction
    center_x = crop.x + crop.size / 2.0
    center_y = crop.y + crop.size / 2.0
    return CropBox(x=center_x - new_size / 2.0, y=center_y - new_size / 2.0, size=new_size)


def crop_position_bounds(size: float, dimension: int, overflow_ratio: float = CROP_OVERFLOW_RATIO) -> Tuple[float, float]:
    """Return the allowed coordinate range for a crop side with optional overflow."""

    overflow = size * overflow_ratio
    minimum = -overflow
    maximum = dimension - size + overflow
    if maximum < minimum:
        maximum = minimum
    return minimum, maximum


def max_crop_size(width: int, height: int) -> float:
    """Return the maximum allowed crop size for a frame."""

    return float(max(width, height))


def normalize_crop_with_overflow(
    width: int,
    height: int,
    crop_box: CropBox,
    overflow_ratio: float = CROP_OVERFLOW_RATIO,
) -> CropBox:
    """Clamp a crop box while allowing configurable overflow beyond the image bounds."""

    size = clamp(crop_box.size, 1.0, max_crop_size(width, height))
    min_x, max_x = crop_position_bounds(size, width, overflow_ratio)
    min_y, max_y = crop_position_bounds(size, height, overflow_ratio)
    x = clamp(crop_box.x, min_x, max_x)
    y = clamp(crop_box.y, min_y, max_y)
    return CropBox(x=x, y=y, size=size)


@dataclass(slots=True)
class ManualCrop:
    """Stores start and end crops for optional motion effects."""

    start: CropBox
    end: CropBox

    def copy(self) -> "ManualCrop":
        return ManualCrop(
            start=CropBox(self.start.x, self.start.y, self.start.size),
            end=CropBox(self.end.x, self.end.y, self.end.size),
        )


@dataclass(slots=True)
class ProcessingOptions:
    input_path: Path
    output_dir: Path
    size: int = 480
    fps: str | float = "keep"
    quality: int = 90
    crf: int = 20
    preset: str = "medium"
    mode: str = "auto"
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None
    min_face: float = 0.1
    face_priority: str = "largest"
    threads: int = 4
    pad: float = 0.0
    image_format: str = "jpg"
    video_ext: str = "mp4"
    keep_audio: bool = True
    log_level: str = "info"
    face_detection_enabled: bool = True
    detection_mode: str = "face"
    motion_enabled: bool = True


class ProgressLogger:
    """Wrapper around tqdm to expose logging-compatible progress reporting."""

    def __init__(self) -> None:
        self._bars: List[tqdm] = []

    def add_bar(self, iterable: Iterable, **kwargs) -> tqdm:
        bar = tqdm(iterable, **kwargs)
        self._bars.append(bar)
        return bar

    def close(self) -> None:
        for bar in self._bars:
            bar.close()
        self._bars.clear()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    return logging.getLogger("memoryball-autocrop")


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def run_ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return json.loads(proc.stdout.decode("utf-8"))


def iter_media_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for root, _dirs, files in os.walk(input_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS or ext in SUPPORTED_VIDEO_EXTS:
                yield Path(root) / file


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTS


def safe_output_path(output_dir: Path, src_path: Path, size: int, image_format: str, video_ext: str) -> Path:
    base = src_path.stem
    suffix = f"_{size}x{size}"
    if is_image(src_path):
        ext = f".{image_format.lower()}"
    else:
        ext = f".{video_ext.lower()}"
    return output_dir / f"{base}{suffix}{ext}"


def already_processed(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def with_margin(box: CropBox, margin: float, width: int, height: int) -> CropBox:
    if margin <= 0:
        return box
    size_increase = box.size * margin
    x = clamp(box.x - size_increase / 2, 0, max(0, width - box.size - size_increase))
    y = clamp(box.y - size_increase / 2, 0, max(0, height - box.size - size_increase))
    size = min(width, height, box.size + size_increase)
    return CropBox(x=x, y=y, size=size)


def setup_environment(options: ProcessingOptions) -> logging.Logger:
    ensure_dir(options.output_dir)
    logger = get_logger(options.log_level)
    if not ffmpeg_available():
        logger.error("ffmpeg/ffprobe nicht gefunden. Bitte installieren und in PATH aufnehmen.")
        sys.exit(1)
    return logger
