"""Frame sampling and face cropping."""

from __future__ import annotations

from typing import Callable, Protocol

import cv2
import numpy as np


class Detector(Protocol):
    def detect_faces(self, image: np.ndarray) -> list[dict]: ...


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """`k` frame numbers spread evenly over the clip, skipping the very first and last frames."""
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    step = n_frames / k
    return [int(step * i + step / 2) for i in range(k)]


def read_frames(path: str, indices: list[int]) -> list[tuple[int, np.ndarray]]:
    """Decode only the wanted frames, in order, as RGB arrays.

    Reading straight through and grabbing frames is faster and more reliable on DFDC files
    than seeking with CAP_PROP_POS_FRAMES.
    """
    wanted = set(indices)
    frames: list[tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(path)
    try:
        last = max(indices) if indices else -1
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append((i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            i += 1
    finally:
        cap.release()
    return frames


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int], margin: float, size: int) -> np.ndarray:
    """Square crop around an (x, y, w, h) box, padded by `margin` of the box size, resized to `size`."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w_img, cx + side / 2))
    y1 = int(min(h_img, cy + side / 2))
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class FaceExtractor:
    """Finds the largest confident face in each sampled frame.

    DFDC clips sometimes show two people, and only one may be manipulated. Taking the largest
    face is a simple rule that matches how most clips are framed; it is a known limitation.
    """

    def __init__(
        self,
        margin: float = 0.25,
        size: int = 224,
        min_confidence: float = 0.9,
        detect_max_side: int = 960,
        detector: Detector | None = None,
        detector_factory: Callable[[], Detector] | None = None,
    ):
        self.margin = margin
        self.size = size
        self.min_confidence = min_confidence
        self.detect_max_side = detect_max_side
        self._detector = detector
        self._factory = detector_factory

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            if self._factory is not None:
                self._detector = self._factory()
            else:
                from mtcnn import MTCNN

                self._detector = MTCNN()
        return self._detector

    def largest_face(self, frame: np.ndarray) -> np.ndarray | None:
        # DFDC frames are 1080p. MTCNN finds faces just as well at half that size and runs
        # about four times faster, so detection runs on a smaller copy and the crop is taken
        # from the full-resolution frame.
        scale = min(1.0, self.detect_max_side / max(frame.shape[:2]))
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = [f for f in self.detector.detect_faces(small) if f.get("confidence", 0) >= self.min_confidence]
        if not faces:
            return None
        best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        x, y, w, h = (v / scale for v in best["box"])
        if w < 20 or h < 20:
            return None
        return crop_face(frame, (int(max(0, x)), int(max(0, y)), int(w), int(h)), self.margin, self.size)

    def from_video(self, path: str, k: int) -> list[tuple[int, np.ndarray]]:
        """Face crops from `k` evenly spaced frames. Frames without a face are skipped."""
        indices = sample_frame_indices(frame_count(path), k)
        crops = []
        for i, frame in read_frames(path, indices):
            face = self.largest_face(frame)
            if face is not None:
                crops.append((i, face))
        return crops
