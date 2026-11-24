# Deepfake Video Detection

Detects face-swap deepfakes in short videos. It samples frames from a clip, finds the face in
each one with MTCNN, scores every face with an EfficientNet-B0 fine-tuned on the
[DFDC](https://ai.meta.com/datasets/dfdc/) dataset, and averages the scores into one verdict
for the video.

This started as my graduation project at the Egyptian Russian University (Faculty of
Artificial Intelligence, 2024), graded Excellent. Version 2 is a rewrite of that project: the
first version was a single Colab notebook, trained a small CNN from scratch and had no proper
test split. The notebook from the original exploration is kept in `notebooks/01_explore_dfdc.ipynb`.

## Results

<!-- Filled from runs/efficientnet_b0/report.md after the Kaggle run. -->
Results on the held-out test videos will be added here from the Kaggle run
(`notebooks/02_train_on_kaggle.ipynb`): AUC, log loss, balanced accuracy and the confusion
matrix, at frame level and at video level.

## How it works

```
video ──► 16 evenly spaced frames ──► MTCNN: largest face per frame ──► 224×224 crop
                                                                            │
verdict ◄── mean of the face scores ◄── EfficientNet-B0 + dense head (P(fake) per face)
```

**Faces, not frames.** Manipulation in DFDC happens on the face, and the rest of the frame is
noise for the model. Crops keep 25% extra margin around the detected box, because blending
artefacts tend to sit on the edge of the swapped face.

**Splitting by source video.** Every fake in DFDC is made from a real clip and shows the same
person in the same room. A random split puts a real clip in training and its fakes in the test
set, and the model can then score well by recognising people and backgrounds. Here a real video
and all the fakes made from it always land in the same split (`metadata.split_by_group`). This
makes the numbers lower and honest.

**Transfer learning in two stages.** The head is trained first on frozen ImageNet features. Then
the backbone from `block6a` onward is unfrozen and trained with a learning rate ten times lower.
Batch-norm layers stay frozen throughout, since batches of 32 faces are too small to keep their
statistics stable.

**Imbalance.** DFDC has about five fakes for every real clip. Training uses inverse-frequency
class weights, and the reported metrics are AUC and balanced accuracy, which a model that
answers "fake" every time cannot game.

**Threshold.** The decision threshold is picked on the validation videos and then applied,
unchanged, to the test videos. Both the 0.5 result and the tuned result are reported.

**Augmentation.** Horizontal flips, JPEG re-compression (quality 60 to 100), and small
brightness and contrast changes. Videos shared online are re-encoded, so the model should not
depend on compression detail. There are no crops or rotations, which would move the face edges.

## Running it

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
```

Download DFDC (the 400-clip sample from the Kaggle competition is enough to start) and point
`data.videos_dirs` in `configs/default.yaml` at the folder that holds `metadata.json`. Several
folders can be listed, one per DFDC part.

```bash
python -m deepfake_detector.extract  --config configs/default.yaml   # split + face crops
python -m deepfake_detector.train    --config configs/default.yaml   # writes runs/.../model.keras
python -m deepfake_detector.evaluate --config configs/default.yaml   # metrics.json, report.md, plots
python -m deepfake_detector.predict  --model runs/efficientnet_b0/model.keras some_clip.mp4
```

Extraction can be stopped and started again; videos that are already done are skipped.
Training on a GPU takes a few minutes for the sample. Without a GPU, use the Kaggle notebook.

**Kaggle.** `notebooks/02_train_on_kaggle.ipynb` runs all of the above on a free Kaggle GPU with
the competition data attached, and zips the model and the results for download.

**Web demo.** `app/app.py` is a Gradio app: upload a clip and it shows the verdict and every face
it scored, with the score for each frame.

```bash
pip install -e ".[app]"
MODEL_PATH=runs/efficientnet_b0/model.keras python app/app.py
```

## Tests

```bash
pytest
```

The tests build a miniature DFDC folder of synthetic clips and run extraction, training,
evaluation and prediction end to end with a stand-in face detector, plus unit tests for the
split, frame sampling, cropping and metrics. They run on every push (`.github/workflows/tests.yml`).

## Project layout

```
configs/                 default.yaml for local runs, kaggle.yaml for the notebook
src/deepfake_detector/
  metadata.py            DFDC metadata, split by source video
  faces.py               frame sampling, MTCNN, face crops
  extract.py             step 1: split and save crops, resumable
  dataset.py             tf.data pipeline, augmentation, class weights
  model.py               EfficientNet-B0 head, two-stage unfreezing
  train.py               step 2
  metrics.py             frame and video scores, threshold choice
  evaluate.py            step 3: metrics, report, plots
  predict.py             score one video from the command line or from code
app/app.py               Gradio demo
notebooks/               original exploration, Kaggle training run
tests/
```

## Limitations

- Only the largest face in a frame is scored. In DFDC clips with two people, sometimes only one
  face is swapped, and if it is the smaller one the clip can be missed.
- The model sees single frames, so it cannot use flicker or unnatural motion between frames.
- DFDC actors were recorded for the dataset. Faces in the wild, other face-swap methods and
  heavy filters will lower accuracy; treat the scores as a screening signal, not proof.

## License

MIT
