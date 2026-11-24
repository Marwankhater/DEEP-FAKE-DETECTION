"""Score a single video.

    python -m deepfake_detector.predict --model runs/efficientnet_b0/model.keras clip.mp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import keras
import numpy as np

from .faces import FaceExtractor


@dataclass
class Verdict:
    prob_fake: float | None
    label: str
    faces: int
    frame_probs: list[float] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)


class VideoScorer:
    def __init__(
        self,
        model: keras.Model | str,
        threshold: float = 0.5,
        frames: int = 16,
        extractor: FaceExtractor | None = None,
    ):
        self.model = keras.models.load_model(model) if isinstance(model, str) else model
        self.image_size = int(self.model.input_shape[1])
        self.threshold = threshold
        self.frames = frames
        self.extractor = extractor or FaceExtractor(size=self.image_size)

    def score_faces(self, faces: list[np.ndarray]) -> np.ndarray:
        batch = np.stack(faces).astype("float32")
        return self.model.predict(batch, verbose=0).ravel()

    def __call__(self, path: str) -> tuple[Verdict, list[np.ndarray]]:
        crops = self.extractor.from_video(path, self.frames)
        if not crops:
            return Verdict(None, "no face found", 0), []
        frame_numbers, faces = zip(*crops)
        probs = self.score_faces(list(faces))
        prob = float(np.mean(probs))
        verdict = Verdict(
            prob_fake=prob,
            label="fake" if prob >= self.threshold else "real",
            faces=len(faces),
            frame_probs=[round(float(p), 4) for p in probs],
            frames=list(frame_numbers),
        )
        return verdict, list(faces)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--model", default="runs/efficientnet_b0/model.keras")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args(argv)

    verdict, _ = VideoScorer(args.model, args.threshold, args.frames)(args.video)
    print(json.dumps(verdict.__dict__, indent=2))


if __name__ == "__main__":
    main()
