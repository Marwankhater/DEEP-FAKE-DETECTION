"""Step 1: split the videos and save face crops to disk.

    python -m deepfake_detector.extract --config configs/default.yaml

Crops go to <faces_dir>/<split>/<video>_<frame>.jpg and every crop is listed in
<faces_dir>/index.csv. The script can be stopped and run again; finished videos are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from .config import Config, load_config
from .faces import FaceExtractor
from .metadata import describe, load_metadata, split_by_group

INDEX_COLUMNS = ["image", "video", "frame", "label", "split", "group"]


def extract(config: Config, extractor: FaceExtractor | None = None, log_every: int = 25) -> pd.DataFrame:
    d = config.data
    faces_dir = Path(d.faces_dir)
    faces_dir.mkdir(parents=True, exist_ok=True)

    videos = split_by_group(load_metadata(d.videos_dirs), **vars(config.split))
    videos.to_csv(faces_dir / "videos.csv", index=False)
    print(describe(videos).to_string(), flush=True)

    index_path = faces_dir / "index.csv"
    index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame(columns=INDEX_COLUMNS)
    done = set(index["video"]) | _empty_videos(faces_dir)

    extractor = extractor or FaceExtractor(d.face_margin, d.image_size, d.min_face_confidence)
    started = time.time()
    todo = videos[~videos["video"].isin(done)]
    for n, row in enumerate(todo.itertuples(index=False), 1):
        out_dir = faces_dir / row.split
        out_dir.mkdir(exist_ok=True)
        new_rows = []
        for frame_no, face in extractor.from_video(row.path, d.frames_per_video):
            image = out_dir / f"{Path(row.video).stem}_{frame_no:04d}.jpg"
            cv2.imwrite(str(image), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            new_rows.append(
                {
                    "image": str(image.relative_to(faces_dir)),
                    "video": row.video,
                    "frame": frame_no,
                    "label": row.label,
                    "split": row.split,
                    "group": row.group,
                }
            )
        if new_rows:
            index = pd.concat([index, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            _mark_empty(faces_dir, row.video)
        if n % log_every == 0 or n == len(todo):
            index.to_csv(index_path, index=False)
            rate = n / max(time.time() - started, 1e-6)
            print(f"{n}/{len(todo)} videos, {len(index)} faces, {rate:.2f} videos/s", flush=True)

    index.to_csv(index_path, index=False)
    return index


def _empty_videos(faces_dir: Path) -> set[str]:
    path = faces_dir / "no_faces.txt"
    return set(path.read_text().split()) if path.exists() else set()


def _mark_empty(faces_dir: Path, video: str) -> None:
    with open(faces_dir / "no_faces.txt", "a") as f:
        f.write(video + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    extract(load_config(args.config))


if __name__ == "__main__":
    main()
