"""Image processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import subprocess
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .face_cropper import FaceCropper
from .utils import (
    CROP_OVERFLOW_RATIO,
    CropBox,
    ManualCrop,
    ProcessingOptions,
    clamp,
    normalize_crop_with_overflow,
    safe_output_path,
    square_size_for_circle,
)

register_heif_opener()


IMAGE_CLIP_DURATION = 5.0
IMAGE_START_HOLD = 0.5
IMAGE_END_HOLD = 0.5
IMAGE_TRANSITION_DURATION = max(0.0, IMAGE_CLIP_DURATION - IMAGE_START_HOLD - IMAGE_END_HOLD)
DEFAULT_IMAGE_FPS = 30.0
@dataclass(slots=True)
class ImageResult:
    source: Path
    target: Path
    processed: bool


def _circle_base_size(width: int, height: int, pad: float = 0.0) -> float:
    diameter = float(min(width, height))
    if pad:
        diameter = max(1.0, diameter * (1 - pad))
    return square_size_for_circle(diameter)


def _center_square(width: int, height: int, pad: float = 0.0) -> CropBox:
    size = _circle_base_size(width, height, pad)
    x = (width - size) / 2
    y = (height - size) / 2
    return CropBox(x=x, y=y, size=size)


def _normalize_crop(
    width: int,
    height: int,
    crop_box: CropBox,
    *,
    allow_overflow: bool = True,
) -> CropBox:
    ratio = CROP_OVERFLOW_RATIO if allow_overflow else 0.0
    return normalize_crop_with_overflow(width, height, crop_box, overflow_ratio=ratio)


def determine_crop_box(
    img: Image.Image,
    options: ProcessingOptions,
    face_cropper: Optional[FaceCropper],
) -> CropBox:
    width, height = img.size
    base_crop = _center_square(width, height, options.pad)
    auto_detected = False
    if options.mode == "center":
        crop_box = base_crop
    elif options.mode == "manual" and None not in (options.crop_x, options.crop_y, options.crop_w, options.crop_h):
        crop_box = CropBox(float(options.crop_x), float(options.crop_y), float(min(options.crop_w, options.crop_h)))
    elif options.face_detection_enabled and face_cropper is not None:
        array = np.array(img.convert("RGB"))
        array_bgr = array[:, :, ::-1]
        detections = face_cropper.detect_subjects(array_bgr)
        crop_box = base_crop
        if detections:
            combined = face_cropper.combine_detections(detections, width, height)
            if combined is not None:
                crop_box = combined
            else:
                target = face_cropper.select_detection(detections, width, height)
                if target is not None:
                    size = target.box.size
                    center_x = target.box.x + size / 2
                    center_y = target.box.y + size / 2
                    crop_box = CropBox(x=center_x - size / 2, y=center_y - size / 2, size=size)
                else:
                    focus = face_cropper.focus_window(detections, width, height, base_crop.size)
                    if focus is not None:
                        crop_box = focus
        auto_detected = True
    else:
        crop_box = base_crop
    allow_overflow = not auto_detected
    return _normalize_crop(width, height, crop_box, allow_overflow=allow_overflow)


def determine_motion_manual(
    img: Image.Image,
    options: ProcessingOptions,
    face_cropper: Optional[FaceCropper],
) -> ManualCrop:
    width, height = img.size
    base_crop = _center_square(width, height, options.pad)
    if not options.motion_enabled:
        normalized = _normalize_crop(width, height, base_crop, allow_overflow=False)
        start = CropBox(normalized.x, normalized.y, normalized.size)
        end = CropBox(normalized.x, normalized.y, normalized.size)
        return ManualCrop(start=start, end=end)
    if not options.face_detection_enabled or face_cropper is None:
        normalized = _normalize_crop(width, height, base_crop, allow_overflow=False)
        start = CropBox(normalized.x, normalized.y, normalized.size)
        end = CropBox(normalized.x, normalized.y, normalized.size)
        return ManualCrop(start=start, end=end)

    array = np.array(img.convert("RGB"))
    array_bgr = array[:, :, ::-1]
    detections = face_cropper.detect_subjects(array_bgr)
    if not detections:
        fallback = determine_crop_box(img, options, face_cropper)
        start = CropBox(fallback.x, fallback.y, fallback.size)
        end = CropBox(fallback.x, fallback.y, fallback.size)
        return ManualCrop(start=start, end=end)

    plan = face_cropper.plan_motion(detections, width, height)
    if plan is None:
        fallback = determine_crop_box(img, options, face_cropper)
        start = CropBox(fallback.x, fallback.y, fallback.size)
        end = CropBox(fallback.x, fallback.y, fallback.size)
        return ManualCrop(start=start, end=end)

    start, end = plan
    start = _normalize_crop(width, height, start, allow_overflow=False)
    end = _normalize_crop(width, height, end, allow_overflow=False)
    start = CropBox(start.x, start.y, start.size)
    end = CropBox(end.x, end.y, end.size)

    direction = getattr(options, "motion_direction", "in") or "in"
    if isinstance(direction, str) and direction.lower() == "out":
        start, end = end, start

    return ManualCrop(start=start, end=end)
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
    total_frames = max(1, int(round(duration * fps)))
    rgb_image = image.convert("RGB")

    def render_crop(crop_box: CropBox) -> np.ndarray:
        cropped = rgb_image.crop(crop_box.as_tuple())
        resized = cropped.resize((target, target), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)

    if not motion_enabled:
        start_hold_frames = total_frames
        end_hold_frames = 0
        motion_frames = 0
    else:
        start_hold_frames = max(1, int(round(IMAGE_START_HOLD * fps)))
        end_hold_frames = max(1, int(round(IMAGE_END_HOLD * fps)))
        desired_motion_frames = max(0, int(round(IMAGE_TRANSITION_DURATION * fps)))
        available = max(0, total_frames - start_hold_frames - end_hold_frames)
        motion_frames = min(available, desired_motion_frames)
        leftover = total_frames - (start_hold_frames + end_hold_frames + motion_frames)
        if leftover > 0:
            motion_frames += leftover
        remainder = total_frames - (start_hold_frames + end_hold_frames + motion_frames)
        if remainder < 0:
            reduction = -remainder
            reduce_start = min(reduction, start_hold_frames - 1)
            start_hold_frames -= reduce_start
            reduction -= reduce_start
            if reduction > 0:
                end_hold_frames = max(0, end_hold_frames - reduction)
            motion_frames = max(0, total_frames - start_hold_frames - end_hold_frames)

    start_frame = render_crop(start)
    end_frame = render_crop(end)

    for _ in range(start_hold_frames):
        yield start_frame

    if motion_enabled and motion_frames > 0:
        steps = motion_frames + 1
        for index in range(motion_frames):
            linear = (index + 1) / steps
            eased = linear * linear * (3 - 2 * linear)
            fraction = clamp(eased, 0.0, 1.0)
            crop = _interpolate_crop(start, end, fraction)
            yield render_crop(crop)
    else:
        for _ in range(max(0, motion_frames)):
            yield start_frame

    for _ in range(end_hold_frames):
        yield end_frame


def _encode_video(
    frames: Iterable[np.ndarray],
    path: Path,
    fps: float,
    size: int,
    options: ProcessingOptions,
) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size}x{size}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for frame in frames:
            if frame.shape != (size, size, 3):  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Ungültige Framegröße {frame.shape}, erwartet {(size, size, 3)}"
                )
            rgb = np.ascontiguousarray(frame)
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fehlgeschlagen für {path}")


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
            start_crop = _normalize_crop(width, height, manual_crop.start, allow_overflow=True)
            end_crop = _normalize_crop(width, height, manual_crop.end, allow_overflow=True)
        else:
            if options.motion_enabled:
                auto_manual = determine_motion_manual(img, options, face_cropper)
                start_crop = _normalize_crop(width, height, auto_manual.start, allow_overflow=False)
                end_crop = _normalize_crop(width, height, auto_manual.end, allow_overflow=False)
            else:
                auto_crop = determine_crop_box(img, options, face_cropper)
                start_crop = _normalize_crop(width, height, auto_crop, allow_overflow=False)
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
    _encode_video(frames, output_path, fps, options.size, options)
    return ImageResult(source=path, target=output_path, processed=True)

