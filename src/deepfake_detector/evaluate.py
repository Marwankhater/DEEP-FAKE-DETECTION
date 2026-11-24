"""Step 3: score the held-out test split.

    python -m deepfake_detector.evaluate --config configs/default.yaml

The decision threshold is picked on the validation split and then applied, unchanged, to the
test split. Results go to <run_dir>/metrics.json, report.md and two plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from .config import Config, load_config
from .dataset import load_index, make_dataset
from .metrics import best_threshold, score, video_scores


def predict_frames(model: keras.Model, frames: pd.DataFrame, image_size: int, batch_size: int) -> pd.DataFrame:
    ds = make_dataset(frames, image_size, batch_size, training=False)
    probs = model.predict(ds, verbose=0).ravel()
    out = frames[["video", "frame", "label"]].copy()
    out["prob"] = probs
    return out


def evaluate(config: Config, model: keras.Model | None = None) -> dict:
    d, t = config.data, config.train
    run_dir = Path(config.run_dir)
    model = model or keras.models.load_model(run_dir / "model.keras")
    index = load_index(d.faces_dir)

    preds = {
        split: predict_frames(model, index[index["split"] == split], d.image_size, t.batch_size)
        for split in ("val", "test")
    }
    videos = {split: video_scores(frames) for split, frames in preds.items()}
    threshold = best_threshold(videos["val"]["label"], videos["val"]["prob"])

    test_frames, test_videos = preds["test"], videos["test"]
    results = {
        "test_frames": score(test_frames["label"], test_frames["prob"]),
        "test_videos": score(test_videos["label"], test_videos["prob"]),
        "test_videos_tuned": score(test_videos["label"], test_videos["prob"], threshold),
        "val_videos": score(videos["val"]["label"], videos["val"]["prob"]),
    }

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    test_videos.to_csv(run_dir / "test_videos.csv", index=False)
    (run_dir / "report.md").write_text(report(results))
    plot(test_videos, results["test_videos_tuned"], run_dir)
    print(report(results))
    return results


def report(results: dict) -> str:
    rows = [
        ("Test, per frame (0.5)", results["test_frames"]),
        ("Test, per video (0.5)", results["test_videos"]),
        (f"Test, per video ({results['test_videos_tuned']['threshold']:.2f}, from val)", results["test_videos_tuned"]),
    ]
    lines = [
        "| | Items | AUC | Log loss | Balanced acc. | Precision (fake) | Recall (fake) | F1 (fake) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows:
        lines.append(
            f"| {name} | {r['count']} | {r['auc']:.3f} | {r['log_loss']:.3f} | {r['balanced_accuracy']:.3f} "
            f"| {r['precision_fake']:.3f} | {r['recall_fake']:.3f} | {r['f1_fake']:.3f} |"
        )
    c = results["test_videos_tuned"]["confusion"]
    lines += [
        "",
        "Confusion matrix, test videos at the tuned threshold:",
        "",
        "| | Predicted real | Predicted fake |",
        "|---|---:|---:|",
        f"| Real | {c['tn']} | {c['fp']} |",
        f"| Fake | {c['fn']} | {c['tp']} |",
        "",
    ]
    return "\n".join(lines)


def plot(videos: pd.DataFrame, tuned: dict, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    y, p = videos["label"].to_numpy(), videos["prob"].to_numpy()
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, color="#1E3A5F", lw=2, label=f"AUC {roc_auc_score(y, p):.3f}")
        ax.plot([0, 1], [0, 1], color="#B0B7C3", lw=1, ls="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, test videos")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(run_dir / "roc.png", dpi=150)
        plt.close(fig)

    c = tuned["confusion"]
    matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Test videos, threshold {tuned['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
