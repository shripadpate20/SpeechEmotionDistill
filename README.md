# 🔊 Speech Emotion Recognition via Knowledge Distillation

This repository demonstrates an efficient and scalable approach to **Speech Emotion Recognition (SER)** using **Knowledge Distillation (KD)**. A large pre-trained **Wav2Vec2** model serves as the teacher, while a lightweight **LiteCNN** model is trained to replicate its performance with significantly fewer parameters and faster inference times.

---

## 🎯 Live Demo

Test the model directly in your browser using our interactive Streamlit app:

🔗 **[🎙️ Speech Emotion Recognition App](https://speechemotiondistill-sjfvberakplyvwznf62bri.streamlit.app/)**

Upload a `.wav` file (16kHz mono) and receive instant emotion predictions.

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [📊 Model Architecture](#-model-architecture)
- [🧠 Knowledge Distillation](#-knowledge-distillation)
- [📁 Dataset](#-dataset)
- [⚙️ Setup](#-setup)
- [🏋️ Training Details](#-training-details)
- [📈 Metrics](#-metrics)
- [🧪 Results](#-results)
- [📦 Outputs](#-outputs)
- [🚀 Future Work](#-future-work)
- [🙌 Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Speech Emotion Recognition is crucial for improving human-computer interaction. While large models like **Wav2Vec2** achieve high accuracy, they are computationally expensive.

This project uses **Knowledge Distillation** to train a **compact CNN model** that:
- Matches the accuracy of large models within a margin
- Is much faster and lighter for real-time inference
- Works directly on raw audio without handcrafted features

---

## 📊 Model Architecture

### 🧠 Teacher: `Wav2Vec2ForSequenceClassification`

- Pre-trained transformer-based model from `facebook/wav2vec2-base`
- Fine-tuned on emotion labels (ANG, DIS, FEA, HAP, SAD, NEU)
- ~95M parameters
- Serves as the reference model during distillation

### 🎓 Student: `LiteCNN`

A custom 1D CNN built for efficiency:

```python
Input: Raw waveform (1D) → Conv1D → Depthwise Separable Conv → Bottleneck → Adaptive Pooling → Fully Connected → Output
