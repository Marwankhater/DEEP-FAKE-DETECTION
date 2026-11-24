from pathlib import Path

import pytest

from conftest import CentreDetector, write_video
from deepfake_detector.config import Config, DataConfig, SplitConfig, TrainConfig
from deepfake_detector.evaluate import evaluate
from deepfake_detector.extract import extract
from deepfake_detector.faces import FaceExtractor
from deepfake_detector.metadata import load_faceforensics, load_sources
from deepfake_detector.train import train

SPLITS = str(Path(__file__).parents[1] / "configs" / "faceforensics_splits")
# Pairs taken from the official splits: two from train, one from val, one from test.
PAIRS = {"train": [("071", "054"), ("087", "081")], "val": [("720", "672")], "test": [("953", "974")]}


@pytest.fixture
def ff_root(tmp_path):
    root = tmp_path / "ff"
    real = root / "original_sequences" / "youtube" / "c23" / "videos"
    real.mkdir(parents=True)
    seed = 0
    for pairs in PAIRS.values():
        for a, b in pairs:
            for i in (a, b):
                write_video(real / f"{i}.mp4", seed=seed); seed += 1
            for method in ("Deepfakes", "FaceSwap"):
                fake = root / "manipulated_sequences" / method / "c23" / "videos"
                fake.mkdir(parents=True, exist_ok=True)
                write_video(fake / f"{a}_{b}.mp4", seed=seed, fake=True); seed += 1
                write_video(fake / f"{b}_{a}.mp4", seed=seed, fake=True); seed += 1
    # Clips that must be ignored: the separate DeepFakeDetection set, and an id outside the splits.
    actors = root / "original_sequences" / "actors" / "c23" / "videos"
    actors.mkdir(parents=True)
    write_video(actors / "01__walking.mp4", seed=99)
    write_video(real / "9999.mp4", seed=98)
    return root


def test_official_splits_and_pairs(ff_root):
    df = load_faceforensics(str(ff_root), SPLITS)
    assert len(df) == 8 * 3  # 8 real ids, each with one fake per method
    assert set(df["method"]) == {"original", "Deepfakes", "FaceSwap"}
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert df.loc[df["video"] == "ff/original/953", "split"].item() == "test"
    assert df.loc[df["video"] == "ff/Deepfakes/054_071", "split"].item() == "train"
    assert not df["video"].str.contains("walking|9999").any()


def test_no_person_is_in_two_splits(ff_root):
    df = load_faceforensics(str(ff_root), SPLITS)
    ids = df["video"].str.split("/").str[-1].str.split("_")
    person_split = {}
    for people, split in zip(ids, df["split"]):
        for p in people:
            assert person_split.setdefault(p, split) == split


def test_train_on_faceforensics_and_test_on_dfdc(ff_root, dfdc_folder, tmp_path):
    config = Config(
        data=DataConfig(
            faces_dir=str(tmp_path / "faces"), frames_per_video=3, image_size=64,
            sources=[
                {"type": "faceforensics", "root": str(ff_root), "splits_dir": SPLITS},
                {"type": "dfdc", "dirs": [str(dfdc_folder)], "role": "cross_test"},
            ],
        ),
        split=SplitConfig(seed=3),
        train=TrainConfig(batch_size=8, epochs_head=1, epochs_finetune=1, weights=None, patience=1),
        run_dir=str(tmp_path / "run"),
    )
    index = extract(config, extractor=FaceExtractor(size=64, detector=CentreDetector()))
    assert set(index["split"]) == {"train", "val", "test", "cross"}
    assert set(index.loc[index["split"] == "cross", "dataset"]) == {"dfdc"}

    train(config, verbose=0)
    results = evaluate(config)
    assert set(results["test_by_method"]) == {"Deepfakes", "FaceSwap"}
    assert results["cross_videos"]["count"] == 24
    report = (tmp_path / "run" / "report.md").read_text()
    assert "Cross-dataset" in report and "Deepfakes" in report


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError):
        load_sources([{"type": "celebdf"}], SplitConfig())
