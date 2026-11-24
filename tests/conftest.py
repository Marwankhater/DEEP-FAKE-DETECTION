import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pytest


class CentreDetector:
    """Stands in for MTCNN: reports one face in the middle of every frame."""

    def __init__(self, confidence=0.99):
        self.confidence = confidence

    def detect_faces(self, image):
        h, w = image.shape[:2]
        return [{"box": [w // 4, h // 4, w // 2, h // 2], "confidence": self.confidence}]


def write_video(path, frames=24, size=96, seed=0, fake=False):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (size, size))
    base = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    for i in range(frames):
        frame = np.roll(base, i, axis=1).copy()
        if fake:
            # A bright square in the face area gives the tiny test model something to learn.
            frame[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 250
        writer.write(frame)
    writer.release()


@pytest.fixture
def dfdc_folder(tmp_path):
    """A miniature DFDC folder: 8 real clips, each with 2 fakes made from it."""
    folder = tmp_path / "videos"
    folder.mkdir()
    meta = {}
    for r in range(8):
        real = f"real{r:02d}.mp4"
        write_video(folder / real, seed=r)
        meta[real] = {"label": "REAL", "split": "train"}
        for f in range(2):
            fake = f"fake{r:02d}_{f}.mp4"
            write_video(folder / fake, seed=100 + r * 2 + f, fake=True)
            meta[fake] = {"label": "FAKE", "split": "train", "original": real}
    meta["missing.mp4"] = {"label": "REAL", "split": "train"}
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def centre_detector():
    return CentreDetector()
