"""Web demo: upload a short video and see the score for each face the model looked at.

    pip install -e ".[app]"
    MODEL_PATH=runs/efficientnet_b0/model.keras python app/app.py

Also runs as a Hugging Face Space; set MODEL_PATH to wherever the model file lives.
"""

from __future__ import annotations

import os

import gradio as gr

from deepfake_detector.predict import VideoScorer

MODEL_PATH = os.environ.get("MODEL_PATH", "runs/efficientnet_b0/model.keras")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

scorer = VideoScorer(MODEL_PATH, threshold=THRESHOLD, frames=16)


def analyse(video_path: str):
    if not video_path:
        return "Upload a video first.", []
    verdict, faces = scorer(video_path)
    if verdict.prob_fake is None:
        return "No face was found in the sampled frames. Try a clip where one face is clearly visible.", []

    summary = (
        f"### {verdict.label.capitalize()}\n"
        f"Average fake score **{verdict.prob_fake:.2f}** over {verdict.faces} faces "
        f"(threshold {THRESHOLD:.2f})."
    )
    gallery = [(face, f"frame {n}: {p:.2f}") for face, n, p in zip(faces, verdict.frames, verdict.frame_probs)]
    return summary, gallery


with gr.Blocks(title="Deepfake video detector") as demo:
    gr.Markdown(
        "# Deepfake video detector\n"
        "EfficientNet-B0 fine-tuned on face crops from the DFDC dataset. It samples 16 frames, "
        "finds the largest face in each with MTCNN, scores every face and averages the scores. "
        "Best on short clips with one clearly visible face. Research demo, not a forensic tool."
    )
    with gr.Row():
        video = gr.Video(label="Video", sources=["upload"])
        with gr.Column():
            result = gr.Markdown()
            faces = gr.Gallery(label="Faces the model scored", columns=4, height="auto")
    gr.Button("Analyse", variant="primary").click(analyse, inputs=video, outputs=[result, faces])

if __name__ == "__main__":
    demo.launch()
