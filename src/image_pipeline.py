"""Image processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .face_cropper import FaceCropper
from .utils import CropBox, ProcessingOptions, clamp, safe_output_path

register_heif_opener()


IMAGE_CLIP_DURATION = 5.0
DEFAULT_IMAGE_FPS = 30.0
MAX_MOTION_FRACTION = 0.05


@dataclass(slots=True)
class ImageResult:
    source: Path
    target: Path
    processed: bool


def _center_square(width: int, height: int, pad: float = 0.0) -> CropBox:
    size = min(width, height)
    if pad:
        size = int(size * (1 - pad))
        size = max(1, size)
    x = (width - size) / 2
    y = (height - size) / 2
    return CropBox(x=x, y=y, size=size)


def _normalize_crop(width: int, height: int, crop_box: CropBox) -> CropBox:
    size = min(crop_box.size, width, height)
    size = max(1, size)
    x = clamp(crop_box.x, 0, max(0, width - size))
    y = clamp(crop_box.y, 0, max(0, height - size))
    return CropBox(x=x, y=y, size=size)


def determine_crop_box(
    img: Image.Image,
    options: ProcessingOptions,
    face_cropper: Optional[FaceCropper],
) -> CropBox:
    width, height = img.size
    base_crop = _center_square(width, height, options.pad)
    if options.mode == "center":
        crop_box = base_crop
    elif options.mode == "manual" and None not in (options.crop_x, options.crop_y, options.crop_w, options.crop_h):
        crop_box = CropBox(float(options.crop_x), float(options.crop_y), float(min(options.crop_w, options.crop_h)))
    elif options.face_detection_enabled and face_cropper is not None:
        array = np.array(img.convert("RGB"))
        array_bgr = array[:, :, ::-1]
        detections = face_cropper.detect_subjects(array_bgr)
        focus = face_cropper.focus_window(detections, width, height, base_crop.size)
        crop_box = focus or base_crop
    else:
        crop_box = base_crop
    return _normalize_crop(width, height, crop_box)


def _apply_crop(img: Image.Image, crop: CropBox) -> Image.Image:
    return img.crop(crop.as_tuple())


def _resize_square(arr: Image.Image, target: int) -> Image.Image:
    return arr.resize((target, target), Image.Resampling.LANCZOS)


def _preferred_fps(options: ProcessingOptions) -> float:
    if options.fps == "keep":
        return DEFAULT_IMAGE_FPS
    try:
        fps = float(options.fps)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return DEFAULT_IMAGE_FPS
    return max(1.0, fps)


def _available_motion_types(slack: int) -> list[str]:
    motions: list[str] = []
    if slack > 0:
        motions.extend(["pan_right", "pan_down"])
    if slack > 1:
        motions.extend(["zoom_in", "zoom_out"])
    return motions


def _select_motion(path: Path, slack: int) -> str:
    motions = _available_motion_types(slack)
    if not motions:
        return "static"
    digest = hashlib.sha1(str(path).encode("utf-8")).digest()
    index = digest[0] % len(motions)
    return motions[index]


def _motion_parameters(base_size: int, target: int, slack: int, motion: str) -> tuple[float, float, float, float, float, float]:
    center = (base_size - target) / 2
    if slack <= 0 or motion == "static":
        return center, center, float(target), center, center, float(target)

    max_shift = max(1.0, min(slack, target * MAX_MOTION_FRACTION))

    if motion == "pan_right":
        start_x = max(0.0, center - max_shift / 2)
        end_x = min(base_size - target, start_x + max_shift)
        start_y = end_y = center
        size_start = size_end = float(target)
    elif motion == "pan_down":
        start_y = max(0.0, center - max_shift / 2)
        end_y = min(base_size - target, start_y + max_shift)
        start_x = end_x = center
        size_start = size_end = float(target)
    else:
        zoom = max(1.0, min(slack, target * MAX_MOTION_FRACTION))
        if motion == "zoom_in":
            size_start = min(base_size, float(target + zoom))
            size_end = float(target)
        else:  # zoom_out
            size_start = float(target)
            size_end = min(base_size, float(target + zoom))
        start_x = (base_size - size_start) / 2
        start_y = (base_size - size_start) / 2
        end_x = (base_size - size_end) / 2
        end_y = (base_size - size_end) / 2

    return start_x, start_y, size_start, end_x, end_y, size_end


def _iter_motion_frames(
    base: Image.Image,
    target: int,
    path: Path,
    fps: float,
    duration: float,
) -> Iterable[np.ndarray]:
    base_size = min(base.size)
    if base_size < target:
        base = _resize_square(base, target)
        base_size = target

    slack = base_size - target
    motion = _select_motion(path, slack)
    start_x, start_y, size_start, end_x, end_y, size_end = _motion_parameters(base_size, target, slack, motion)

    frames = max(1, int(round(duration * fps)))
    base_rgb = base.convert("RGB")

    for index in range(frames):
        if frames == 1:
            t = 0.0
        else:
            t = index / (frames - 1)
        size = size_start + (size_end - size_start) * t
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t

        size = clamp(size, 1, base_size)
        x = clamp(x, 0, base_size - size)
        y = clamp(y, 0, base_size - size)

        crop_box = (
            int(round(x)),
            int(round(y)),
            int(round(x + size)),
            int(round(y + size)),
        )
        cropped = base_rgb.crop(crop_box)
        frame = cropped.resize((target, target), Image.Resampling.LANCZOS)
        array = np.array(frame)
        yield cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _write_video(frames: Iterable[np.ndarray], path: Path, fps: float, size: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():  # pragma: no cover - defensive
        raise RuntimeError(f"Kann Videodatei nicht schreiben: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def process_image(
    path: Path,
    options: ProcessingOptions,
    face_cropper: Optional[FaceCropper],
    manual_crop: Optional[CropBox] = None,
) -> ImageResult:
    output_path = safe_output_path(options.output_dir, path, options.size, options.image_format, options.video_ext)
    video_suffix = f".{options.video_ext.lower()}"
    if output_path.suffix.lower() != video_suffix:
        output_path = output_path.with_suffix(video_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
        return ImageResult(source=path, target=output_path, processed=False)

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        if manual_crop is not None:
            crop_box = _normalize_crop(width, height, manual_crop)
        else:
            crop_box = determine_crop_box(img, options, face_cropper)

        cropped = _apply_crop(img, crop_box)

    fps = _preferred_fps(options)
    frames = _iter_motion_frames(cropped, options.size, path, fps, IMAGE_CLIP_DURATION)
    _write_video(frames, output_path, fps, options.size)
    return ImageResult(source=path, target=output_path, processed=True)

