"""EfficientNet-B0 with a small binary head.

Training happens in two stages. First only the head learns, on top of frozen ImageNet
features. Then the last blocks of the backbone are unfrozen and trained with a lower learning
rate. Batch-norm layers stay frozen in both stages: with batches of 32 faces their running
statistics would drift away from the ImageNet ones and hurt more than help.
"""

from __future__ import annotations

import keras
from keras import layers


def build_model(image_size: int = 224, dropout: float = 0.3, weights: str | None = "imagenet") -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(image_size, image_size, 3), pooling="avg"
    )
    base.trainable = False

    inputs = keras.Input((image_size, image_size, 3), name="face")
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    return keras.Model(inputs, outputs, name="deepfake_efficientnet_b0")


def backbone(model: keras.Model) -> keras.Model:
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def unfreeze_from(model: keras.Model, layer_name: str) -> int:
    """Make every backbone layer from `layer_name` onward trainable, except batch norm.

    Returns how many layers were unfrozen.
    """
    base = backbone(model)
    names = [layer.name for layer in base.layers]
    if layer_name not in names:
        raise ValueError(f"{layer_name!r} is not a layer of {base.name}")
    start = names.index(layer_name)

    base.trainable = True
    unfrozen = 0
    for i, layer in enumerate(base.layers):
        trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        unfrozen += trainable
    return unfrozen


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )
