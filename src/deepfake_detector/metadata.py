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
