# 🕵️ Deepfake Detection System

A deep learning system that detects manipulated images and videos using **MTCNN** for face detection and **CNN models** built with TensorFlow/Keras — achieving high precision in identifying deepfake content.

---

## 📌 Overview

With the rapid rise of AI-generated media, detecting deepfakes has become a critical challenge. This project tackles the problem end-to-end — from face extraction to classification — using a robust deep learning pipeline trained and validated to distinguish real from manipulated content.

---

## ✨ Features

- 🎯 **Precise Face Detection** — MTCNN extracts faces accurately from images and video frames
- 🧠 **CNN-Based Classification** — Custom-trained model classifies real vs. deepfake content
- 🔄 **Data Augmentation** — Expands training diversity to improve generalization
- ⚙️ **Optimized Pipeline** — Full preprocessing, training, validation, and testing workflow
- 📈 **High Precision Results** — Achieved strong detection accuracy on test data

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Face Detection | MTCNN |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, OpenCV |
| Visualization | Matplotlib |

---

## 📁 Project Structure

```
deepfake-detection/
│
├── data/
│   ├── real/                  # Real face samples
│   └── fake/                  # Deepfake face samples
│
├── src/
│   ├── face_extractor.py      # MTCNN face detection & cropping
│   ├── preprocessing.py       # Augmentation & normalization
│   ├── model.py               # CNN architecture definition
│   ├── train.py               # Training & validation loop
│   └── predict.py             # Inference on new inputs
│
├── models/
│   └── deepfake_detector.h5   # Saved trained model
│
├── notebooks/
│   └── analysis.ipynb         # EDA & results visualization
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

### Run Detection

```bash
# Train the model
python src/train.py

# Run prediction on an image
python src/predict.py --input path/to/image.jpg
```

---

## 🔬 How It Works

```
Input Image / Video
       │
       ▼
 MTCNN Face Detection
       │
       ▼
 Face Cropping & Preprocessing
       │
       ▼
 Data Augmentation
       │
       ▼
 CNN Classification Model
       │
       ▼
 Output: REAL or FAKE ✅ / ❌
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | High |
| Precision | High |
| Validation Loss | Low |

> Detailed metrics and confusion matrix available in `notebooks/analysis.ipynb`

---

## 🎓 Project Context

This project was developed as a **Graduation Project** at the **Egyptian Russian University**, Faculty of Artificial Intelligence — demonstrating practical application of deep learning in media forensics.

---

## 🙋‍♂️ Author

**Marwan Ashraf**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/marwan-ashraf-811660219/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://www.linkedin.com/in/marwan-ashraf-811660219/)

