"""Step 1: split the videos and save face crops to disk.

    python -m deepfake_detector.extract --config configs/default.yaml

Crops go to <faces_dir>/<split>/<video>_<frame>.jpg and every crop is listed in
<faces_dir>/index.csv. The script can be stopped and run again; finished videos are skipped.
While the detector works on one clip, the next clips are already being decoded in background
threads, so the GPU does not wait for the disk.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from .config import Config, load_config
from .faces import FaceExtractor, frame_count, read_frames, sample_frame_indices
from .metadata import describe, load_metadata, load_sources, split_by_group

INDEX_COLUMNS = ["image", "video", "frame", "label", "split", "group", "method", "dataset"]


def video_table(config: Config) -> pd.DataFrame:
    d = config.data
    if d.sources:
        return load_sources(d.sources, config.split)
    videos = split_by_group(load_metadata(d.videos_dirs), **vars(config.split))
    videos["method"], videos["dataset"] = "dfdc", "dfdc"
    return videos


def _decode(path: str, k: int):
    return read_frames(path, sample_frame_indices(frame_count(path), k))


def _decoded_in_order(pool: ThreadPoolExecutor, paths, k: int, ahead: int):
    """Decoded frames for each path, in order, with at most `ahead` clips decoded in advance.

    Executor.map would queue every clip at once and hold thousands of decoded frames in memory.
    """
    paths = list(paths)
    pending = deque(pool.submit(_decode, p, k) for p in paths[:ahead])
    for i in range(len(paths)):
        frames = pending.popleft().result()
        if i + ahead < len(paths):
            pending.append(pool.submit(_decode, paths[i + ahead], k))
        yield frames


def extract(config: Config, extractor: FaceExtractor | None = None, log_every: int = 25, workers: int = 4) -> pd.DataFrame:
    d = config.data
    faces_dir = Path(d.faces_dir)
    faces_dir.mkdir(parents=True, exist_ok=True)

    videos = video_table(config)
    videos.to_csv(faces_dir / "videos.csv", index=False)
    print(describe(videos).to_string(), flush=True)

    index_path = faces_dir / "index.csv"
    index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame(columns=INDEX_COLUMNS)
    done = set(index["video"]) | _empty_videos(faces_dir)

    extractor = extractor or FaceExtractor(d.face_margin, d.image_size, d.min_face_confidence, d.detect_max_side)
    todo = videos[~videos["video"].isin(done)].reset_index(drop=True)
    started = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decoded = _decoded_in_order(pool, todo["path"], d.frames_per_video, ahead=workers * 2)
        for n, (row, frames) in enumerate(zip(todo.itertuples(index=False), decoded), 1):
            out_dir = faces_dir / row.split
            out_dir.mkdir(exist_ok=True)
            found = False
            stem = row.video.replace("/", "__")
            for frame_no, face in extractor.from_frames(frames):
                image = out_dir / f"{Path(stem).stem}_{frame_no:04d}.jpg"
                cv2.imwrite(str(image), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                found = True
                rows.append(
                    {
                        "image": str(image.relative_to(faces_dir)),
                        "video": row.video,
                        "frame": frame_no,
                        "label": row.label,
                        "split": row.split,
                        "group": row.group,
                        "method": row.method,
                        "dataset": row.dataset,
                    }
                )
            if not found:
                _mark_empty(faces_dir, row.video)
            if n % log_every == 0 or n == len(todo):
                index = pd.concat([index, pd.DataFrame(rows, columns=INDEX_COLUMNS)], ignore_index=True)
                rows = []
                index.to_csv(index_path, index=False)
                rate = n / max(time.time() - started, 1e-6)
                left = (len(todo) - n) / max(rate, 1e-6) / 60
                print(f"{n}/{len(todo)} videos, {len(index)} faces, {rate:.2f} videos/s, about {left:.0f} min left", flush=True)

    if rows:
        index = pd.concat([index, pd.DataFrame(rows, columns=INDEX_COLUMNS)], ignore_index=True)
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
