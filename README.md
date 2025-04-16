# 🎧 Speech Emotion Recognition with Knowledge Distillation

This repository implements **Speech Emotion Recognition (SER)** using a compact student CNN model trained via **Knowledge Distillation** from a larger pre-trained **Wav2Vec2.0 teacher model**. The goal is to compress a large model into a lightweight one while maintaining performance — ideal for deployment in resource-constrained environments like mobile devices.

## 🔗 Live Demo

👉 [Try the model on Streamlit!](https://speechemotiondistill-sjfvberakplyvwznf62bri.streamlit.app/)

---

## 🧠 Key Concepts

- **Teacher Model**: `Wav2Vec2ForSequenceClassification` (from HuggingFace Transformers)
- **Student Model**: Custom lightweight `LiteCNN` architecture using depthwise separable convolutions and bottleneck layers.
- **Knowledge Distillation**:
  - Trains the student using both **hard labels** (ground truth) and **soft labels** (teacher's output probabilities).
  - Helps the student learn better generalization with fewer parameters.

---

## 📁 Dataset: CREMA-D

- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **Classes**: `ANG`, `DIS`, `FEA`, `HAP`, `SAD`, `NEU`
- Audio files are preprocessed to 16 kHz and trimmed/padded to 1 second.

> 💡 Ensure you download the dataset and set `AUDIO_DIR` in your script accordingly.

---

## 🏗️ Model Architecture

### 🧑‍🏫 Teacher: Wav2Vec2

- HuggingFace's `facebook/wav2vec2-base`
- Fine-tuned on emotion classification (6 labels)

### 🧑‍🎓 Student: LiteCNN

```python
Input: [B, 1, 16000]
→ Conv1D + ReLU + MaxPool
→ Depthwise + Pointwise Conv
→ Bottleneck Conv
→ AdaptiveAvgPool
→ Flatten → Fully Connected → Output (6 classes)
```

- Lightweight
- Depthwise separable convolutions reduce complexity
- 25x–30x fewer parameters than the teacher

---

## 🧪 Training Strategy

- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (adaptive learning rate)
- **Loss Function**: Custom Distillation Loss combining:
  - `CrossEntropyLoss` (hard labels)
  - `KL Divergence` (soft labels)
- **Alpha**: Weighting between task and distillation loss
- **Temperature**: Softens logits for distillation

---

## 📊 Performance Metrics

During training and validation:
- **Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **Loss Components**: Total, Task, Distillation

---

## 🚀 How to Run

### 🔧 1. Install Dependencies

```bash
pip install torch torchaudio scikit-learn tqdm transformers
```

### 📦 2. Prepare Dataset

- Download the **CREMA-D** dataset.
- Set `AUDIO_DIR` to your local path.

### 🏋️ 3. Train the Student Model

```bash
python train_student_kd.py
```

This script:
- Loads teacher model weights
- Trains student with knowledge distillation
- Logs metrics
- Saves `crema_student_model_kd.pth`

---

## 💻 Deployment

The trained student model is deployed via **Streamlit** for real-time inference.

🔗 **[Launch Streamlit App](https://speechemotiondistill-sjfvberakplyvwznf62bri.streamlit.app/)**  
Upload an audio clip and see the model predict the emotion in seconds.

---

## 📈 Model Compression Summary

| Model         | Parameters | Size     | Accuracy (Val) | Notes                        |
|---------------|------------|----------|----------------|------------------------------|
| Teacher (Wav2Vec2) | ~95M       | Large    | High           | Pre-trained transformer model |
| Student (LiteCNN)  | ~3.5M      | Small    | Slight drop    | Optimized for speed & memory  |

🧊 **Compression Ratio**: ~27x smaller!

---

## 🧪 Example Outputs

| Audio Input       | Predicted Emotion |
|-------------------|-------------------|
| happy_01.wav      | `HAP`             |
| sad_03.wav        | `SAD`             |
| angry_07.wav      | `ANG`             |

---

## 🙌 Acknowledgements

- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)

---

## 📬 Contact

For questions or improvements, feel free to reach out or open an issue.  
Happy experimenting! 🎉
