"""Image processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .face_cropper import FaceCropper
from .utils import CropBox, ManualCrop, ProcessingOptions, clamp, safe_output_path

register_heif_opener()


IMAGE_CLIP_DURATION = 5.0
DEFAULT_IMAGE_FPS = 30.0
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
def _preferred_fps(options: ProcessingOptions) -> float:
    if options.fps == "keep":
        return DEFAULT_IMAGE_FPS
    try:
        fps = float(options.fps)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return DEFAULT_IMAGE_FPS
    return max(1.0, fps)


def _interpolate_crop(start: CropBox, end: CropBox, fraction: float) -> CropBox:
    fraction = clamp(fraction, 0.0, 1.0)
    x = start.x + (end.x - start.x) * fraction
    y = start.y + (end.y - start.y) * fraction
    size = start.size + (end.size - start.size) * fraction
    return CropBox(x=x, y=y, size=size)


def _iter_motion_frames(
    image: Image.Image,
    start: CropBox,
    end: CropBox,
    target: int,
    fps: float,
    duration: float,
    motion_enabled: bool,
) -> Iterable[np.ndarray]:
    frames = max(1, int(round(duration * fps)))
    rgb_image = image.convert("RGB")

    for index in range(frames):
        if motion_enabled and frames > 1:
            fraction = index / (frames - 1)
        else:
            fraction = 0.0
        crop = _interpolate_crop(start, end, fraction)
        cropped = rgb_image.crop(crop.as_tuple())
        resized = cropped.resize((target, target), Image.Resampling.LANCZOS)
        frame_array = np.array(resized)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        yield frame_bgr


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
    manual_crop: Optional[ManualCrop] = None,
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
            start_crop = _normalize_crop(width, height, manual_crop.start)
            end_crop = _normalize_crop(width, height, manual_crop.end)
        else:
            auto_crop = determine_crop_box(img, options, face_cropper)
            start_crop = _normalize_crop(width, height, auto_crop)
            end_crop = start_crop

        if not options.motion_enabled:
            start_crop = CropBox(end_crop.x, end_crop.y, end_crop.size)

    fps = _preferred_fps(options)
    frames = _iter_motion_frames(
        img,
        start_crop,
        end_crop,
        options.size,
        fps,
        IMAGE_CLIP_DURATION,
        options.motion_enabled,
    )
    _write_video(frames, output_path, fps, options.size)
    return ImageResult(source=path, target=output_path, processed=True)

