"""tf.data pipelines over the face crops written by `extract`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_index(faces_dir: str) -> pd.DataFrame:
    index = pd.read_csv(Path(faces_dir) / "index.csv")
    index["filepath"] = [str(Path(faces_dir) / p) for p in index["image"]]
    return index


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights. DFDC has roughly five fakes for every real clip."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    if (counts == 0).any():
        return {0: 1.0, 1: 1.0}
    total = counts.sum()
    return {i: float(total / (2 * counts[i])) for i in range(2)}


def _augment(image: tf.Tensor) -> tf.Tensor:
    # Compression and colour changes are what real-world uploads do to a face, so the model
    # should not lean on them. No crops or rotations: blending artefacts sit near the face edge.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_jpeg_quality(image, 60, 100)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, 20.0)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0)


def make_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int = 42,
) -> tf.data.Dataset:
    """Batches of (image in 0..255 float32, label). EfficientNet rescales inside the model."""
    paths = frame["filepath"].to_numpy()
    labels = frame["label"].to_numpy().astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (image_size, image_size), antialias=True)
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
        image = _augment(image) if training else tf.cast(image, tf.float32)
        return image, label

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
