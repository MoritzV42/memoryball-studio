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
        max_tracking_gap: int = 15,
        max_step_fraction: float = 0.15,
    ) -> None:
        self.min_face = min_face
        self.face_priority = face_priority
        self.smoother = ExponentialSmoother(alpha=smoothing)
        self.max_tracking_gap = max(0, int(max_tracking_gap))
        self.max_step_fraction = max(0.0, float(max_step_fraction))
        self._last_smoothed: Optional[CropBox] = None
        self._lost_frames = 0

        self._use_mediapipe = mp is not None
        self._face_detection: Optional["mp.solutions.face_detection.FaceDetection"] = None
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None
        self._upper_body_cascade: Optional[cv2.CascadeClassifier] = None
        self._full_body_cascade: Optional[cv2.CascadeClassifier] = None
        self._hog: Optional[cv2.HOGDescriptor] = None
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
            self._profile_cascade = self._load_optional_cascade("haarcascade_profileface.xml")
            self._upper_body_cascade = self._load_optional_cascade("haarcascade_upperbody.xml")
            self._full_body_cascade = self._load_optional_cascade("haarcascade_fullbody.xml")

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._saliency = None
        if hasattr(cv2, "saliency") and hasattr(cv2.saliency, "StaticSaliencyFineGrained_create"):
            with contextlib.suppress(Exception):
                self._saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    def close(self) -> None:
        if self._use_mediapipe and self._face_detection is not None:
            with contextlib.suppress(Exception):
                self._face_detection.close()

    def _load_optional_cascade(self, filename: str) -> Optional[cv2.CascadeClassifier]:
        path = cv2.data.haarcascades + filename
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():  # pragma: no cover - optional assets may be missing
            return None
        return cascade

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

        def run_detector(
            cascade: Optional[cv2.CascadeClassifier],
            *,
            weight: float,
            scale_factor: float = 1.1,
            min_neighbors: int = 5,
            flip: bool = False,
        ) -> None:
            if cascade is None:
                return
            source = gray
            if flip:
                source = cv2.flip(gray, 1)
            results = cascade.detectMultiScale(
                source,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, width, height) in results:
                if flip:
                    x = w - (x + width)
                size = max(width, height)
                min_size = self.min_face * min(w, h)
                size = max(size, min_size)
                cx = x + width / 2
                cy = y + height / 2
                x_adj = clamp(cx - size / 2, 0, max(0, w - size))
                y_adj = clamp(cy - size / 2, 0, max(0, h - size))
                base = CropBox(x=x_adj, y=y_adj, size=size)
                score = min(1.0, weight + min(0.4, size / max(1.0, min(w, h)) * 0.4))
                detections.append(
                    DetectionResult(score=score, box=self._circle_aligned_box(base, w, h))
                )

        run_detector(self._cascade, weight=0.95)
        run_detector(self._profile_cascade, weight=0.75)
        run_detector(self._profile_cascade, weight=0.75, flip=True)
        run_detector(self._upper_body_cascade, weight=0.65, scale_factor=1.05, min_neighbors=3)
        run_detector(self._full_body_cascade, weight=0.6, scale_factor=1.02, min_neighbors=3)
        return detections

    def _detect_people(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        if self._hog is None:
            return detections
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = self._hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        min_size = self.min_face * min(w, h)
        center_x = w / 2
        center_y = h / 2
        max_radius = math.hypot(center_x, center_y)
        for (x, y, width, height), raw_score in zip(rects, weights):
            size = max(width, height, min_size)
            cx = x + width / 2
            cy = y + height / 2
            x_adj = clamp(cx - size / 2, 0, max(0, w - size))
            y_adj = clamp(cy - size / 2, 0, max(0, h - size))
            base = CropBox(x=x_adj, y=y_adj, size=size)
            distance = math.hypot(cx - center_x, cy - center_y)
            center_bonus = max(0.0, 1.0 - distance / max(1.0, max_radius))
            score = float(max(0.2, min(1.0, raw_score * 0.7 + center_bonus * 0.5)))
            detections.append(
                DetectionResult(score=score, box=self._circle_aligned_box(base, w, h))
            )
        return detections

    def _detect_saliency(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        h, w = image.shape[:2]
        saliency_map: Optional[np.ndarray] = None
        if self._saliency is not None:
            success, saliency = self._saliency.computeSaliency(image)
            if success and saliency is not None:
                saliency_map = (saliency * 255).astype("uint8")
        if saliency_map is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            lap = cv2.Laplacian(blurred, cv2.CV_32F)
            saliency_map = cv2.convertScaleAbs(lap)
        blurred_saliency = cv2.GaussianBlur(saliency_map, (5, 5), 0)
        _threshold, binary = cv2.threshold(
            blurred_saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return detections
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        image_area = float(w * h)
        center_x = w / 2
        center_y = h / 2
        max_radius = math.hypot(center_x, center_y)
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.02:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            size = max(width, height)
            cx = x + width / 2
            cy = y + height / 2
            x_adj = clamp(cx - size / 2, 0, max(0, w - size))
            y_adj = clamp(cy - size / 2, 0, max(0, h - size))
            base = CropBox(x=x_adj, y=y_adj, size=max(size, self.min_face * min(w, h)))
            distance = math.hypot(cx - center_x, cy - center_y)
            center_bonus = max(0.0, 1.0 - distance / max(1.0, max_radius))
            size_ratio = min(1.0, size / max(1.0, min(w, h)))
            interest = 0.25 + center_bonus * 0.5 + size_ratio * 0.35
            detections.append(
                DetectionResult(score=float(min(0.9, interest)), box=self._circle_aligned_box(base, w, h))
            )
        return detections

    @staticmethod
    def _square_iou(a: CropBox, b: CropBox) -> float:
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.size, b.x + b.size)
        y2 = min(a.y + a.size, b.y + b.size)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = a.size * a.size + b.size * b.size - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _merge_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        if not detections:
            return []
        merged: List[DetectionResult] = []
        for det in sorted(detections, key=lambda d: (d.score, d.box.size), reverse=True):
            duplicate = False
            for kept in merged:
                if self._square_iou(det.box, kept.box) >= 0.4:
                    duplicate = True
                    if det.score > kept.score:
                        kept.score = det.score
                        kept.box = det.box
                    break
            if not duplicate:
                merged.append(DetectionResult(score=det.score, box=CropBox(det.box.x, det.box.y, det.box.size)))
        return merged

    def _circle_aligned_box(self, box: CropBox, width: int, height: int) -> CropBox:
        expanded = expand_crop_for_circle(box)
        min_size = max(box.size, self.min_face * min(width, height))
        max_size = max_crop_size(width, height)
        size = clamp(expanded.size, min_size, max_size)
        center_x = expanded.x + expanded.size / 2
        center_y = expanded.y + expanded.size / 2
        x = clamp(center_x - size / 2, 0.0, max(0.0, width - size))
        y = clamp(center_y - size / 2, 0.0, max(0.0, height - size))
        return CropBox(x=x, y=y, size=size)

    def _filter_relevant_detections(
        self,
        detections: List[DetectionResult],
        width: int,
        height: int,
    ) -> List[DetectionResult]:
        if not detections:
            return []
        max_size = max((det.box.size for det in detections), default=0.0)
        if max_size <= 0:
            return []

        def center_factor(det: DetectionResult) -> float:
            cx = det.box.x + det.box.size / 2
            cy = det.box.y + det.box.size / 2
            norm_dx = abs(cx - width / 2) / max(width / 2, 1.0)
            norm_dy = abs(cy - height / 2) / max(height / 2, 1.0)
            return max(0.0, 1.0 - 0.5 * (norm_dx + norm_dy))

        size_threshold = max_size * 0.4
        filtered = [
            det
            for det in detections
            if det.box.size >= size_threshold or det.score >= 0.6 or center_factor(det) >= 0.65
        ]

        def priority(det: DetectionResult) -> float:
            size_ratio = det.box.size / max(1.0, max_size)
            return det.score * 0.6 + center_factor(det) * 0.3 + size_ratio * 0.4

        ranked_pool = filtered if filtered else detections
        ranked = sorted(ranked_pool, key=priority, reverse=True)[:5]

        if len(ranked) == 1 and len(detections) > 1:
            alt = max((det for det in detections if det not in ranked), key=priority, default=None)
            if alt is not None:
                ranked.append(alt)
        return ranked

    def combine_detections(
        self,
        detections: List[DetectionResult],
        width: int,
        height: int,
    ) -> Optional[CropBox]:
        relevant = self._filter_relevant_detections(detections, width, height)
        if len(relevant) < 2:
            return None
        min_x = min(det.box.x for det in relevant)
        max_x = max(det.box.x + det.box.size for det in relevant)
        min_y = min(det.box.y for det in relevant)
        max_y = max(det.box.y + det.box.size for det in relevant)
        cover_width = max_x - min_x
        cover_height = max_y - min_y
        min_face_size = self.min_face * min(width, height)
        largest_size = max(det.box.size for det in relevant)
        min_size = max(min_face_size, largest_size)
        max_size = max_crop_size(width, height)
        size = max(cover_width, cover_height, largest_size)
        size = clamp(size * 1.05, min_size, max_size)

        weights = [max(0.1, det.score) * det.box.size for det in relevant]
        total_weight = sum(weights) or float(len(relevant))
        centers_x = [(det.box.x + det.box.size / 2) for det in relevant]
        centers_y = [(det.box.y + det.box.size / 2) for det in relevant]
        weighted_cx = sum(cx * w for cx, w in zip(centers_x, weights)) / total_weight
        weighted_cy = sum(cy * w for cy, w in zip(centers_y, weights)) / total_weight

        min_allowed_x = max_x - size
        max_allowed_x = min_x
        min_allowed_y = max_y - size
        max_allowed_y = min_y

        x = clamp(weighted_cx - size / 2, min_allowed_x, max_allowed_x)
        y = clamp(weighted_cy - size / 2, min_allowed_y, max_allowed_y)

        return CropBox(x=x, y=y, size=size)

    def detect_subjects(self, image: np.ndarray) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        if self._use_mediapipe:
            detections.extend(self._detect_with_mediapipe(image))
        else:
            detections.extend(self._detect_with_cascade(image))
        best_score = max((det.score for det in detections), default=0.0)
        extra_people: List[DetectionResult] = []
        if best_score < 0.6 or len(detections) < 1:
            extra_people = self._detect_people(image)
            if extra_people:
                detections.extend(extra_people)
        if not detections and extra_people:
            detections = extra_people
        if not detections:
            detections = self._detect_saliency(image)
        return self._merge_detections(detections)

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

    def _position_with_constraints(
        self,
        preferred: float,
        min_edge: float,
        max_edge: float,
        size: float,
        total: int,
    ) -> float:
        limit = max(0.0, total - size)
        coverage_min = max(max_edge - size, 0.0)
        coverage_max = min(min_edge, limit)
        if coverage_min > coverage_max:
            return clamp(preferred, 0.0, limit)
        return clamp(preferred, coverage_min, coverage_max)

    def _crop_from_bounds(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        size: float,
        width: int,
        height: int,
        top_ratio: float,
    ) -> CropBox:
        preferred_x = (min_x + max_x) / 2 - size / 2
        preferred_y = min_y - size * top_ratio
        x = self._position_with_constraints(preferred_x, min_x, max_x, size, width)
        y = self._position_with_constraints(preferred_y, min_y, max_y, size, height)
        return CropBox(x=x, y=y, size=size)

    def plan_motion(
        self,
        detections: List[DetectionResult],
        width: int,
        height: int,
        *,
        body_scale: float = 1.7,
        face_margin: float = 0.12,
        body_top_ratio: float = 0.2,
        face_top_ratio: float = 0.08,
    ) -> Optional[tuple[CropBox, CropBox]]:
        relevant = self._filter_relevant_detections(detections, width, height)
        if not relevant:
            return None
        relevant = sorted(relevant, key=lambda det: det.box.size, reverse=True)

        min_x = min(det.box.x for det in relevant)
        max_x = max(det.box.x + det.box.size for det in relevant)
        min_y = min(det.box.y for det in relevant)
        max_y = max(det.box.y + det.box.size for det in relevant)

        min_face_size = self.min_face * min(width, height)
        largest_size = max(det.box.size for det in relevant)
        min_size = max(min_face_size, largest_size)
        max_size = max_crop_size(width, height)

        span_width = max_x - min_x
        span_height = max_y - min_y
        base_span = max(span_width, span_height, largest_size)
        face_size = clamp(base_span * (1.0 + face_margin), min_size, max_size)
        body_size = clamp(face_size * body_scale, min_size, max_size)

        fits_all = span_width <= max_size and span_height <= max_size

        if not fits_all and len(relevant) >= 2:
            primary = relevant[0].box
            secondary = relevant[1].box
            primary_min = max(min_face_size, primary.size)
            secondary_min = max(min_face_size, secondary.size)
            start = self._crop_from_bounds(
                primary.x,
                primary.x + primary.size,
                primary.y,
                primary.y + primary.size,
                clamp(primary.size * body_scale, primary_min, max_size),
                width,
                height,
                body_top_ratio,
            )
            end = self._crop_from_bounds(
                secondary.x,
                secondary.x + secondary.size,
                secondary.y,
                secondary.y + secondary.size,
                clamp(secondary.size * (1.0 + face_margin), secondary_min, max_size),
                width,
                height,
                face_top_ratio,
            )
            return start, end

        start = self._crop_from_bounds(min_x, max_x, min_y, max_y, body_size, width, height, body_top_ratio)
        end = self._crop_from_bounds(min_x, max_x, min_y, max_y, face_size, width, height, face_top_ratio)
        return start, end

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
        combined = self.combine_detections(detections, width, height)
        if combined is not None:
            window_size = combined.size
            focus = combined
        else:
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
