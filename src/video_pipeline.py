"""Video processing pipeline."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .face_cropper import FaceCropper
from .utils import CropBox, ProcessingOptions, clamp, run_ffprobe, safe_output_path


@dataclass(slots=True)
class VideoResult:
    source: Path
    target: Path
    processed: bool


def _center_crop(width: int, height: int, pad: float = 0.0) -> CropBox:
    size = min(width, height)
    if pad:
        size = int(size * (1 - pad))
        size = max(1, size)
    x = (width - size) / 2
    y = (height - size) / 2
    return CropBox(x=x, y=y, size=size)


def _compute_crop(frame: np.ndarray, options: ProcessingOptions, face_cropper: Optional[FaceCropper], fallback: CropBox) -> CropBox:
    height, width = frame.shape[:2]
    if options.mode == "center":
        crop = fallback
    elif options.mode == "manual" and None not in (options.crop_x, options.crop_y, options.crop_w, options.crop_h):
        crop = CropBox(float(options.crop_x), float(options.crop_y), float(min(options.crop_w, options.crop_h)))
    elif options.face_detection_enabled and face_cropper is not None:
        crop = face_cropper.track(frame, width, height, fallback)
    else:
        crop = fallback
    crop.size = min(crop.size, width, height)
    crop.x = clamp(crop.x, 0, width - crop.size)
    crop.y = clamp(crop.y, 0, height - crop.size)
    return crop


def _iter_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def process_video(path: Path, options: ProcessingOptions, face_cropper: Optional[FaceCropper]) -> VideoResult:
    output_path = safe_output_path(options.output_dir, path, options.size, options.image_format, options.video_ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
        return VideoResult(source=path, target=output_path, processed=False)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if options.fps != "keep":
        fps_out = float(options.fps)
    else:
        fps_out = fps

    fallback = _center_crop(width, height, options.pad)

    ffprobe_data = run_ffprobe(path)
    has_audio = any(stream.get("codec_type") == "audio" for stream in ffprobe_data.get("streams", []))

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{options.size}x{options.size}",
        "-r",
        f"{fps_out}",
        "-i",
        "pipe:0",
        "-map",
        "1:v:0",
    ]
    if options.keep_audio and has_audio:
        ffmpeg_cmd.extend(["-map", "0:a:0"])
        if options.fps == "keep":
            ffmpeg_cmd.extend(["-c:a", "copy"])
        else:
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        ffmpeg_cmd.append("-an")
    ffmpeg_cmd.extend([
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps_out}",
        str(output_path),
    ])

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    for frame in _iter_frames(cap):
        crop_box = _compute_crop(frame, options, face_cropper, fallback)
        x1, y1, x2, y2 = crop_box.as_tuple()
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (options.size, options.size), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    cap.release()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fehlgeschlagen für {path}")

    return VideoResult(source=path, target=output_path, processed=True)
