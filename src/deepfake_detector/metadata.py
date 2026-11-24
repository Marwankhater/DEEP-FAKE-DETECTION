"""Reading DFDC metadata and splitting it without leaking faces between splits.

Every fake in DFDC is made from a real video, and it shows the same person in the same
room. If the real clip lands in training and its fakes land in the test set, the model can
score well by recognising people and backgrounds instead of manipulation. So the split is
done on groups: a real video and all the fakes made from it always stay together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"REAL": 0, "FAKE": 1}


def load_metadata(videos_dirs: list[str] | str) -> pd.DataFrame:
    """One row per video that exists on disk: video, path, label (1 = fake), original, group."""
    if isinstance(videos_dirs, (str, Path)):
        videos_dirs = [videos_dirs]

    rows = []
    for folder in map(Path, videos_dirs):
        meta_path = folder / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {folder}")
        meta = json.loads(meta_path.read_text())
        for name, info in meta.items():
            path = folder / name
            if not path.exists():
                continue
            label = LABELS[info["label"]]
            original = info.get("original") or name
            rows.append(
                {
                    "video": name,
                    "path": str(path),
                    "label": label,
                    "original": original,
                    "group": original if label == 1 else name,
                }
            )

    if not rows:
        raise ValueError(f"No videos found in {list(map(str, videos_dirs))}")
    return pd.DataFrame(rows).drop_duplicates("video").reset_index(drop=True)


def split_by_group(
    df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> pd.DataFrame:
    """Add a `split` column (train / val / test). Groups never cross splits.

    Groups are shuffled and handed out until each split holds roughly its share of videos.
    """
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be below 1")

    sizes = df.groupby("group").size()
    groups = sizes.index.to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    total = len(df)
    targets = {"test": test_fraction * total, "val": val_fraction * total}
    assigned: dict[str, str] = {}
    counts = {"test": 0, "val": 0}
    for group in groups:
        for split in ("test", "val"):
            if counts[split] < targets[split]:
                assigned[group] = split
                counts[split] += sizes[group]
                break
        else:
            assigned[group] = "train"

    out = df.copy()
    out["split"] = out["group"].map(assigned)
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Videos per split and label, for a quick sanity check."""
    table = df.pivot_table(index="split", columns="label", values="video", aggfunc="count", fill_value=0)
    return table.rename(columns={0: "real", 1: "fake"})


# FaceForensics++ ---------------------------------------------------------------------------
#
# Real clips are named by an id ("071.mp4"). A fake is named target_source ("071_054.mp4"): the
# face of 054 pasted onto video 071. The official splits list pairs of ids that are only ever
# swapped with each other, so a pair is a natural group: both real clips and both fakes made
# from them stay in one split, and no person appears in two splits.

FF_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


def _ff_method(path: Path, methods: tuple[str, ...]) -> str | None:
    """'original' for real clips, the manipulation name for fakes, None for anything else."""
    parts = [p.lower() for p in path.parts]
    if any(p.startswith("original") or p in ("youtube", "real") for p in parts[:-1]):
        if any(p in ("actors", "deepfakedetection") for p in parts):
            return None
        return "original"
    for m in methods:
        if m.lower() in parts:
            return m
    return None


def load_faceforensics(root: str, splits_dir: str, methods: tuple[str, ...] = FF_METHODS) -> pd.DataFrame:
    """Every FaceForensics++ clip under `root`, with its official split."""
    split_of: dict[str, str] = {}
    pair_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for a, b in json.loads((Path(splits_dir) / f"{split}.json").read_text()):
            key = "ff:" + "_".join(sorted((a, b)))
            for i in (a, b):
                split_of[i], pair_of[i] = split, key

    rows = []
    for path in sorted(Path(root).rglob("*.mp4")):
        method = _ff_method(path.relative_to(root), methods)
        if method is None:
            continue
        stem = path.stem
        ids = stem.split("_")
        if method == "original":
            if len(ids) != 1 or ids[0] not in split_of:
                continue
            target, label = ids[0], 0
        else:
            if len(ids) != 2 or ids[0] not in split_of:
                continue
            target, label = ids[0], 1
        rows.append(
            {
                "video": f"ff/{method}/{stem}",
                "path": str(path),
                "label": label,
                "original": f"ff/original/{target}",
                "group": pair_of[target],
                "method": method,
                "dataset": "faceforensics",
                "split": split_of[target],
            }
        )
    if not rows:
        raise ValueError(f"No FaceForensics++ clips found under {root}")
    return pd.DataFrame(rows).drop_duplicates("video").reset_index(drop=True)


def load_sources(sources: list[dict], split) -> pd.DataFrame:
    """Combine datasets described in the config into one table with a `split` column.

    Each source is a dict with `type` (dfdc or faceforensics) and an optional `role`:
      train_test (default): the dataset is split and used for training and testing.
      cross_test: every clip goes to the `cross` split. The model never trains on it, which
                  is how generalisation to a new dataset is measured.
    """
    frames = []
    for src in sources:
        kind, role = src["type"], src.get("role", "train_test")
        if kind == "faceforensics":
            df = load_faceforensics(src["root"], src["splits_dir"], tuple(src.get("methods", FF_METHODS)))
        elif kind == "dfdc":
            df = load_metadata(src["dirs"])
            df["method"], df["dataset"] = "dfdc", "dfdc"
            df["video"] = "dfdc/" + df["video"]
            df["group"] = "dfdc:" + df["group"]
            df["original"] = "dfdc/" + df["original"]
            if role != "cross_test":
                df = split_by_group(df, split.val_fraction, split.test_fraction, split.seed)
        else:
            raise ValueError(f"Unknown source type {kind!r}")
        if role == "cross_test":
            df["split"] = "cross"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
