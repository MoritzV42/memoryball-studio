"""Face detection and smoothing utilities."""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import List, Optional

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - better error hint for Windows users
    raise RuntimeError(
        "OpenCV (cv2) ist nicht installiert. Bitte `pip install -r requirements.txt` ausführen."
    ) from exc
import numpy as np

_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - optional dependency check
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    mp = None
    _IMPORT_ERROR = exc

from .utils import CropBox, clamp, expand_crop_for_circle, max_crop_size


@dataclass(slots=True)
class DetectionResult:
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
    def __init__(
        self,
        min_face: float = 0.1,
        face_priority: str = "largest",
        smoothing: float = 0.4,
        mode: str = "face",
        max_tracking_gap: int = 15,
        max_step_fraction: float = 0.15,
    ) -> None:
        self.mode = mode
        self.min_face = min_face
        self.face_priority = face_priority
        self.smoother = ExponentialSmoother(alpha=smoothing)
        self.max_tracking_gap = max(0, int(max_tracking_gap))
        self.max_step_fraction = max(0.0, float(max_step_fraction))
        self._last_smoothed: Optional[CropBox] = None
        self._lost_frames = 0

        self._use_mediapipe = self.mode == "face" and mp is not None
        self._face_detection: Optional["mp.solutions.face_detection.FaceDetection"] = None
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._hog: Optional[cv2.HOGDescriptor] = None

        if self.mode == "face":
            if self._use_mediapipe:
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.4
                )
            else:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
                if self._cascade.empty():  # pragma: no cover - depends on local OpenCV data files
                    message = (
                        "mediapipe ist nicht installiert und das OpenCV-Haar-Cascade-Modell fehlt."
                        " Bitte installiere entweder mediapipe oder stelle sicher, dass OpenCV korrekt"
                        " installiert ist."
                    )
                    if mp is None:
                        raise RuntimeError(message) from _IMPORT_ERROR
                    raise RuntimeError(message)
        elif self.mode == "person":
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            raise ValueError(f"Unbekannter Erkennungsmodus: {mode}")

    def close(self) -> None:
        if self._use_mediapipe and self._face_detection is not None:
            with contextlib.suppress(Exception):
                self._face_detection.close()

    def _mp_box_to_crop(
        self,
        detection: "mp.framework.formats.detection_pb2.Detection",
        width: int,
        height: int,
    ) -> CropBox:
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
        base = CropBox(x=x, y=y, size=size)
        return self._circle_aligned_box(base, width, height)

    def _detect_with_mediapipe(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        h, w = image.shape[:2]
        assert self._face_detection is not None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)
        if results.detections:
            for detection in results.detections:
                score = detection.score[0] if detection.score else 0.0
                box = self._mp_box_to_crop(detection, w, h)
                detections.append(DetectionResult(score=score, box=box))
        return detections

    def _detect_with_cascade(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        h, w = image.shape[:2]
        assert self._cascade is not None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, width, height) in faces:
            size = max(width, height)
            size = max(size, self.min_face * min(w, h))
            x_adj = clamp(x + width / 2 - size / 2, 0, max(0, w - size))
            y_adj = clamp(y + height / 2 - size / 2, 0, max(0, h - size))
            base = CropBox(x=x_adj, y=y_adj, size=size)
            detections.append(DetectionResult(score=1.0, box=self._circle_aligned_box(base, w, h)))
        return detections

    def _detect_people(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        if self._hog is None:
            return detections
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = self._hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        min_size = self.min_face * min(w, h)
        for (x, y, width, height), score in zip(rects, weights):
            size = max(width, height, min_size)
            cx = x + width / 2
            cy = y + height / 2
            x_adj = clamp(cx - size / 2, 0, max(0, w - size))
            y_adj = clamp(cy - size / 2, 0, max(0, h - size))
            base = CropBox(x=x_adj, y=y_adj, size=size)
            detections.append(DetectionResult(score=float(score), box=self._circle_aligned_box(base, w, h)))
        return detections

    def _circle_aligned_box(self, box: CropBox, width: int, height: int) -> CropBox:
        expanded = expand_crop_for_circle(box)
        min_size = max(box.size, self.min_face * min(width, height))
        max_size = max_crop_size(width, height)
        size = clamp(expanded.size, min_size, max_size)
        center_x = expanded.x + expanded.size / 2
        center_y = expanded.y + expanded.size / 2
        x = center_x - size / 2
        y = center_y - size / 2
        return CropBox(x=x, y=y, size=size)

    def detect_subjects(self, image: np.ndarray) -> List[DetectionResult]:
        if self.mode == "person":
            return self._detect_people(image)
        if self._use_mediapipe:
            return self._detect_with_mediapipe(image)
        return self._detect_with_cascade(image)

    def select_detection(self, detections: List[DetectionResult], width: int, height: int) -> Optional[DetectionResult]:
        if not detections:
            return None
        if self.face_priority == "center":
            center_x = width / 2
            center_y = height / 2
            return min(
                detections,
                key=lambda det: (det.box.x + det.box.size / 2 - center_x) ** 2
                + (det.box.y + det.box.size / 2 - center_y) ** 2,
            )
        if self.face_priority == "largest" or self.face_priority == "all":
            return max(detections, key=lambda det: det.box.size)
        return detections[0]

    def _axis_spans(
        self,
        detections: List[DetectionResult],
        axis: str,
        total: int,
        window_size: float,
    ) -> List[tuple[float, float, float]]:
        spans: List[tuple[float, float, float]] = []
        if not detections:
            return spans
        max_size = max((d.box.size for d in detections), default=1.0)
        axis_center = total / 2
        for det in detections:
            start = det.box.x if axis == "x" else det.box.y
            end = start + det.box.size
            if self.face_priority == "largest":
                weight = 1.0 + det.box.size / max(1.0, max_size)
            elif self.face_priority == "center":
                det_center = (start + end) / 2
                distance = abs(det_center - axis_center)
                normalized = distance / max(axis_center, 1.0)
                weight = 1.0 + max(0.0, 1.0 - normalized)
            else:  # "all"
                overlap_ratio = min(det.box.size, window_size) / max(window_size, 1.0)
                weight = 1.0 + overlap_ratio
            spans.append((start, end, weight))
        return spans

    def _best_axis_position(
        self,
        spans: List[tuple[float, float, float]],
        window_size: float,
        total: int,
    ) -> float:
        limit = max(0.0, total - window_size)
        if limit <= 0:
            return 0.0
        candidates = {0.0, limit}
        center = clamp(total / 2 - window_size / 2, 0.0, limit)
        candidates.add(center)
        for start, end, _weight in spans:
            candidates.add(clamp(start, 0.0, limit))
            candidates.add(clamp(end - window_size, 0.0, limit))

        def coverage(pos: float) -> float:
            window_end = pos + window_size
            value = 0.0
            for start, end, weight in spans:
                overlap = min(window_end, end) - max(pos, start)
                if overlap > 0:
                    value += overlap * weight
            return value

        best_score = -1.0
        best_pos = center
        for pos in sorted(candidates):
            score = coverage(pos)
            if score > best_score or (score == best_score and abs(pos - center) < abs(best_pos - center)):
                best_score = score
                best_pos = pos
        return clamp(best_pos, 0.0, limit)

    def focus_window(
        self,
        detections: List[DetectionResult],
        width: int,
        height: int,
        window_size: float,
    ) -> Optional[CropBox]:
        if not detections:
            return None
        spans_x = self._axis_spans(detections, "x", width, window_size)
        spans_y = self._axis_spans(detections, "y", height, window_size)
        x = self._best_axis_position(spans_x, window_size, width) if spans_x else clamp((width - window_size) / 2, 0.0, max(0.0, width - window_size))
        y = self._best_axis_position(spans_y, window_size, height) if spans_y else clamp((height - window_size) / 2, 0.0, max(0.0, height - window_size))
        return CropBox(x=x, y=y, size=window_size)

    def track(self, frame: np.ndarray, width: int, height: int, fallback: CropBox) -> CropBox:
        detections = self.detect_subjects(frame)
        window_size = fallback.size
        focus: Optional[CropBox]
        target = self.select_detection(detections, width, height)
        if target is not None:
            window_size = target.box.size
            focus = CropBox(target.box.x, target.box.y, target.box.size)
        else:
            focus = self.focus_window(detections, width, height, window_size)
        if focus is None:
            self._lost_frames += 1
            if self._last_smoothed is not None and self._lost_frames <= self.max_tracking_gap:
                window_size = self._last_smoothed.size
                preserved = CropBox(self._last_smoothed.x, self._last_smoothed.y, window_size)
                preserved.x = clamp(preserved.x, 0, max(0, width - window_size))
                preserved.y = clamp(preserved.y, 0, max(0, height - window_size))
                return preserved
            self._last_smoothed = None
            self._lost_frames = 0
            self.smoother.reset()
            return fallback

        self._lost_frames = 0
        smoothed = self.smoother.update(focus)

        if self._last_smoothed is not None and self.max_step_fraction > 0:
            max_step = window_size * self.max_step_fraction
            dx = smoothed.x - self._last_smoothed.x
            dy = smoothed.y - self._last_smoothed.y
            if abs(dx) > max_step:
                smoothed.x = self._last_smoothed.x + math.copysign(max_step, dx)
            if abs(dy) > max_step:
                smoothed.y = self._last_smoothed.y + math.copysign(max_step, dy)

        smoothed.size = window_size
        smoothed.x = clamp(smoothed.x, 0, max(0, width - window_size))
        smoothed.y = clamp(smoothed.y, 0, max(0, height - window_size))
        self._last_smoothed = CropBox(smoothed.x, smoothed.y, smoothed.size)
        return self._last_smoothed
