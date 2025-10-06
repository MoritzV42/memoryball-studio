"""Utility functions for memoryball-autocrop."""
from __future__ import annotations

import json
import logging
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
        return int(self.x), int(self.y), int(self.x + self.size), int(self.y + self.size)


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
