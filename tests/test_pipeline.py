"""End to end on a miniature dataset: extract, train, evaluate, predict.

The model starts from random weights and trains for two short epochs on 64 px crops, so this
checks that the steps fit together, not that the model is any good.
"""

import json

from conftest import CentreDetector
from deepfake_detector.config import Config, DataConfig, SplitConfig, TrainConfig
from deepfake_detector.evaluate import evaluate
from deepfake_detector.extract import extract
from deepfake_detector.faces import FaceExtractor
from deepfake_detector.model import build_model, unfreeze_from
from deepfake_detector.predict import VideoScorer
from deepfake_detector.train import train


def tiny_config(videos, tmp_path):
    return Config(
        data=DataConfig(videos_dirs=[str(videos)], faces_dir=str(tmp_path / "faces"), frames_per_video=4, image_size=64),
        split=SplitConfig(val_fraction=0.25, test_fraction=0.25, seed=3),
        train=TrainConfig(batch_size=8, epochs_head=1, epochs_finetune=1, weights=None, patience=1),
        run_dir=str(tmp_path / "run"),
    )


def test_unfreezing_leaves_batch_norm_frozen():
    model = build_model(image_size=64, weights=None)
    n = unfreeze_from(model, "block6a_expand_conv")
    assert n > 0
    base = model.layers[1]
    assert not any(l.trainable for l in base.layers if l.__class__.__name__ == "BatchNormalization")
    assert model.output_shape == (None, 1)


def test_full_pipeline(dfdc_folder, tmp_path):
    config = tiny_config(dfdc_folder, tmp_path)
    extractor = FaceExtractor(size=64, detector=CentreDetector())

    index = extract(config, extractor=extractor)
    assert len(index) == 24 * 4
    assert (index.groupby("group")["split"].nunique() == 1).all()

    again = extract(config, extractor=extractor)
    assert len(again) == len(index), "a second run must not duplicate faces"

    model_path = train(config, verbose=0)
    assert model_path.exists()

    results = evaluate(config)
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["test_videos"]["count"] == results["test_videos"]["count"] > 0
    assert (tmp_path / "run" / "report.md").exists()
    assert (tmp_path / "run" / "confusion.png").exists()

    scorer = VideoScorer(str(model_path), frames=4, extractor=extractor)
    verdict, faces = scorer(str(dfdc_folder / "fake00_0.mp4"))
    assert verdict.label in {"real", "fake"}
    assert verdict.faces == len(faces) == 4
    assert 0.0 <= verdict.prob_fake <= 1.0
