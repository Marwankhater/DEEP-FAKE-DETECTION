import numpy as np
import pandas as pd

from deepfake_detector.dataset import class_weights
from deepfake_detector.metrics import best_threshold, score, video_scores


def test_video_score_is_the_mean_of_its_frames():
    frames = pd.DataFrame(
        {"video": ["a", "a", "b", "b", "b"], "label": [1, 1, 0, 0, 0], "prob": [0.9, 0.7, 0.1, 0.2, 0.3]}
    )
    v = video_scores(frames).set_index("video")
    assert np.isclose(v.loc["a", "prob"], 0.8)
    assert v.loc["b", "faces"] == 3


def test_score_on_a_perfect_split():
    r = score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r["auc"] == 1.0
    assert r["balanced_accuracy"] == 1.0
    assert r["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_threshold_is_chosen_for_balanced_accuracy():
    labels = [0, 0, 0, 1, 1, 1]
    probs = [0.55, 0.6, 0.62, 0.7, 0.8, 0.9]
    t = best_threshold(labels, probs)
    assert 0.62 < t <= 0.7
    assert score(labels, probs, t)["balanced_accuracy"] == 1.0


def test_class_weights_favour_the_rarer_class():
    w = class_weights(np.array([1, 1, 1, 1, 0]))
    assert w[0] > w[1]
    assert class_weights(np.array([1, 1]))[0] == 1.0
