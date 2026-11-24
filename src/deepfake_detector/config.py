from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    # One or more DFDC folders, each with a metadata.json next to its .mp4 files.
    videos_dirs: list[str] = field(default_factory=lambda: ["data/train_sample_videos"])
    faces_dir: str = "data/faces"
    frames_per_video: int = 16
    face_margin: float = 0.25
    min_face_confidence: float = 0.90
    image_size: int = 224
    detect_max_side: int = 960
    # Optional list of datasets; see metadata.load_sources. When empty, videos_dirs is used.
    sources: list[dict] = field(default_factory=list)


@dataclass
class SplitConfig:
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs_head: int = 3
    epochs_finetune: int = 10
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    unfreeze_from: str = "block6a_expand_conv"
    dropout: float = 0.3
    weights: str | None = "imagenet"
    patience: int = 3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_dir: str = "runs/efficientnet_b0"

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML config. Missing keys fall back to the defaults above."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=_build(DataConfig, raw.get("data")),
        split=_build(SplitConfig, raw.get("split")),
        train=_build(TrainConfig, raw.get("train")),
        run_dir=raw.get("run_dir", Config.run_dir),
    )


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
