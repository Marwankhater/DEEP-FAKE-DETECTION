"""Frame sampling and face cropping."""

from __future__ import annotations

from typing import Callable, Protocol

import cv2
import numpy as np


class Detector(Protocol):
    def detect_faces(self, image): ...


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """`k` frame numbers spread evenly over the clip, skipping the very first and last frames."""
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    step = n_frames / k
    return [int(step * i + step / 2) for i in range(k)]


def read_frames(path: str, indices: list[int], seek_gap: int = 40) -> list[tuple[int, np.ndarray]]:
    """Decode only the wanted frames, in order, as RGB arrays.

    Short gaps are walked with grab(), which is cheap. Long gaps jump with CAP_PROP_POS_FRAMES,
    so a 500-frame clip does not have to be decoded in full to get 8 frames from it. If a jump
    fails to land, the rest is read straight through.
    """
    frames: list[tuple[int, np.ndarray]] = []
    if not indices:
        return frames
    cap = cv2.VideoCapture(path)
    try:
        pos = 0
        for target in sorted(set(indices)):
            if target - pos > seek_gap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == target:
                    pos = target
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            while pos < target:
                if not cap.grab():
                    return frames
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if not ok:
                return frames
            frames.append((target, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
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

    All frames of a clip go to the detector in one batch, which on a GPU is several times
    faster than one call per frame.
    """

    def __init__(
        self,
        margin: float = 0.25,
        size: int = 224,
        min_confidence: float = 0.9,
        detect_max_side: int = 960,
        detector: Detector | None = None,
        detector_factory: Callable[[], Detector] | None = None,
        batch: bool = True,
    ):
        self.margin = margin
        self.size = size
        self.min_confidence = min_confidence
        self.detect_max_side = detect_max_side
        self.batch = batch
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

    def _scale(self, frame: np.ndarray) -> float:
        # DFDC frames are 1080p. MTCNN finds faces just as well on a smaller copy and runs much
        # faster, so detection runs on the copy and the crop comes from the full frame.
        return min(1.0, self.detect_max_side / max(frame.shape[:2]))

    @staticmethod
    def _shrink(frame: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _detect(self, images):
        try:
            return self.detector.detect_faces(images)
        except ValueError:
            # mtcnn 1.0 on Keras 3 raises when its last stage receives zero candidate boxes,
            # which simply means there is no face in this frame.
            return None

    def _crop_best(self, frame: np.ndarray, detected, scale: float) -> np.ndarray | None:
        faces = [f for f in (detected or []) if f.get("confidence", 0) >= self.min_confidence]
        if not faces:
            return None
        best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        x, y, w, h = (v / scale for v in best["box"])
        if w < 20 or h < 20:
            return None
        return crop_face(frame, (int(max(0, x)), int(max(0, y)), int(w), int(h)), self.margin, self.size)

    def largest_face(self, frame: np.ndarray) -> np.ndarray | None:
        scale = self._scale(frame)
        return self._crop_best(frame, self._detect(self._shrink(frame, scale)), scale)

    def _one_by_one(self, frames):
        crops = ((i, self.largest_face(f)) for i, f in frames)
        return [(i, c) for i, c in crops if c is not None]

    def from_frames(self, frames: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
        """Face crops for already decoded frames. Frames without a face are skipped."""
        if not frames:
            return []
        if not self.batch or len({f.shape for _, f in frames}) != 1:
            return self._one_by_one(frames)
        scale = self._scale(frames[0][1])
        detected = self._detect([self._shrink(f, scale) for _, f in frames])
        if detected is None or len(detected) != len(frames) or (detected and isinstance(detected[0], dict)):
            # A batch can fail on one bad frame; fall back to one call per frame.
            return self._one_by_one(frames)
        out = []
        for (i, frame), found in zip(frames, detected):
            crop = self._crop_best(frame, found, scale)
            if crop is not None:
                out.append((i, crop))
        return out

    def from_video(self, path: str, k: int) -> list[tuple[int, np.ndarray]]:
        """Face crops from `k` evenly spaced frames. Frames without a face are skipped."""
        return self.from_frames(read_frames(path, sample_frame_indices(frame_count(path), k)))
