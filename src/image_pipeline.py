"""Image processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .face_cropper import FaceCropper
from .utils import CropBox, ProcessingOptions, clamp, safe_output_path

register_heif_opener()


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
    if options.mode == "center":
        crop_box = _center_square(width, height, options.pad)
    elif options.mode == "manual" and None not in (options.crop_x, options.crop_y, options.crop_w, options.crop_h):
        crop_box = CropBox(float(options.crop_x), float(options.crop_y), float(min(options.crop_w, options.crop_h)))
    elif options.face_detection_enabled and face_cropper is not None:
        array = np.array(img.convert("RGB"))
        array_bgr = array[:, :, ::-1]
        detections = face_cropper.detect_faces(array_bgr)
        box = face_cropper.select_face(detections, width, height)
        if box is None:
            crop_box = _center_square(width, height, options.pad)
        else:
            crop_box = box
    else:
        crop_box = _center_square(width, height, options.pad)
    return _normalize_crop(width, height, crop_box)


def _apply_crop(img: Image.Image, crop: CropBox) -> Image.Image:
    return img.crop(crop.as_tuple())


def _resize_square(arr: Image.Image, target: int) -> Image.Image:
    return arr.resize((target, target), Image.Resampling.LANCZOS)


def process_image(
    path: Path,
    options: ProcessingOptions,
    face_cropper: Optional[FaceCropper],
    manual_crop: Optional[CropBox] = None,
) -> ImageResult:
    output_path = safe_output_path(options.output_dir, path, options.size, options.image_format, options.video_ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
        return ImageResult(source=path, target=output_path, processed=False)

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    width, height = img.size

    if manual_crop is not None:
        crop_box = _normalize_crop(width, height, manual_crop)
    else:
        crop_box = determine_crop_box(img, options, face_cropper)

    cropped = _apply_crop(img, crop_box)
    resized = _resize_square(cropped, options.size)

    save_kwargs = {}
    format_name = options.image_format.upper()
    if format_name == "JPG":
        format_name = "JPEG"
    if format_name in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = options.quality
    resized.save(output_path, format=format_name, **save_kwargs)
    return ImageResult(source=path, target=output_path, processed=True)
