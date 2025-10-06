"""Face detection and smoothing utilities."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency check
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    mp = None
    _IMPORT_ERROR = exc

from .utils import CropBox, clamp


@dataclass(slots=True)
class FaceDetectionResult:
    score: float
    box: CropBox


class ExponentialSmoother:
    """Simple exponential moving average smoother for bounding boxes."""

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, box: CropBox) -> CropBox:
        vector = np.array([box.x, box.y, box.size], dtype=np.float32)
        if self._state is None:
            self._state = vector
        else:
            self._state = self.alpha * vector + (1 - self.alpha) * self._state
        return CropBox(float(self._state[0]), float(self._state[1]), float(self._state[2]))


class FaceCropper:
    def __init__(self, min_face: float = 0.1, face_priority: str = "largest", smoothing: float = 0.6) -> None:
        if mp is None:
            raise RuntimeError("mediapipe ist nicht installiert. Bitte `pip install mediapipe` ausführen") from _IMPORT_ERROR
        self._face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        self.min_face = min_face
        self.face_priority = face_priority
        self.smoother = ExponentialSmoother(alpha=smoothing)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._face_detection.close()

    def _mp_box_to_crop(self, detection: "mp.framework.formats.detection_pb2.Detection", width: int, height: int) -> CropBox:
        location = detection.location_data
        relative = location.relative_bounding_box
        w = relative.width * width
        h = relative.height * height
        size = max(w, h)
        size = max(size, self.min_face * min(width, height))
        cx = (relative.xmin + relative.width / 2) * width
        cy = (relative.ymin + relative.height / 2) * height
        x = clamp(cx - size / 2, 0, max(0, width - size))
        y = clamp(cy - size / 2, 0, max(0, height - size))
        size = min(size, width, height)
        return CropBox(x=x, y=y, size=size)

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)
        detections: List[FaceDetectionResult] = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                score = detection.score[0] if detection.score else 0.0
                box = self._mp_box_to_crop(detection, w, h)
                detections.append(FaceDetectionResult(score=score, box=box))
        return detections

    def select_face(self, detections: List[FaceDetectionResult], width: int, height: int) -> Optional[CropBox]:
        if not detections:
            return None
        if self.face_priority == "center":
            cx, cy = width / 2, height / 2
            detections.sort(key=lambda d: (abs((d.box.x + d.box.size / 2) - cx) + abs((d.box.y + d.box.size / 2) - cy)))
        elif self.face_priority == "all":
            # merge by encompassing box
            xs = [d.box.x for d in detections]
            ys = [d.box.y for d in detections]
            max_size = max(d.box.size for d in detections)
            min_x = max(0.0, min(xs))
            min_y = max(0.0, min(ys))
            box = CropBox(x=min_x, y=min_y, size=max_size)
            return box
        else:
            detections.sort(key=lambda d: d.box.size, reverse=True)
        return detections[0].box

    def track(self, frame: np.ndarray, width: int, height: int, fallback: CropBox) -> CropBox:
        detections = self.detect_faces(frame)
        box = self.select_face(detections, width, height)
        if box is None:
            self.smoother.reset()
            return fallback
        smoothed = self.smoother.update(box)
        smoothed.x = clamp(smoothed.x, 0, max(0, width - smoothed.size))
        smoothed.y = clamp(smoothed.y, 0, max(0, height - smoothed.size))
        smoothed.size = min(smoothed.size, width, height)
        return smoothed
