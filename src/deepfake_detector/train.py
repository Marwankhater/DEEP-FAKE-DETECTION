"""Step 2: train on the face crops.

    python -m deepfake_detector.train --config configs/default.yaml

Writes the best model, the training log and a copy of the config to `run_dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras

from .config import Config, load_config, save_config
from .dataset import class_weights, load_index, make_dataset
from .model import build_model, compile_model, unfreeze_from


def train(config: Config, verbose: int = 2) -> Path:
    d, t = config.data, config.train
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    keras.utils.set_random_seed(config.split.seed)
    index = load_index(d.faces_dir)
    train_df = index[index["split"] == "train"]
    val_df = index[index["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise ValueError("Both the train and val splits need faces. Run the extract step first.")

    train_ds = make_dataset(train_df, d.image_size, t.batch_size, training=True, seed=config.split.seed)
    val_ds = make_dataset(val_df, d.image_size, t.batch_size, training=False)
    weights = class_weights(train_df["label"].to_numpy())
    print(f"train faces {len(train_df)}, val faces {len(val_df)}, class weights {weights}", flush=True)

    model_path = run_dir / "model.keras"
    log = keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=False)

    model = build_model(d.image_size, t.dropout, t.weights)
    compile_model(model, t.lr_head)
    model.fit(train_ds, validation_data=val_ds, epochs=t.epochs_head, class_weight=weights, callbacks=[log], verbose=verbose)

    unfrozen = unfreeze_from(model, t.unfreeze_from)
    print(f"fine-tuning {unfrozen} backbone layers from {t.unfreeze_from}", flush=True)
    compile_model(model, t.lr_finetune)
    callbacks = [
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_auc", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=t.patience, restore_best_weights=True),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=t.epochs_head,
        epochs=t.epochs_head + t.epochs_finetune,
        class_weight=weights,
        callbacks=callbacks,
        verbose=verbose,
    )
    if not model_path.exists():
        model.save(model_path)
    return model_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    print(train(load_config(args.config)))


if __name__ == "__main__":
    main()
