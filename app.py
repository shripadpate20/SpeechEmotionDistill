import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np

# Define the model class again
class LiteCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, groups=32, padding=1),  # Depthwise
            nn.Conv1d(64, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Instantiate and load model
model = LiteCNN()
state_dict = torch.load("/content/crema_student_model_kd.pth", map_location=torch.device('cpu'), weights_only=True)

model.load_state_dict(state_dict)
model.eval()

# Emotion labels
emotion_labels = ["ANG", "DIS", "FEA", "HAP", "SAD", "NEU"]

# Preprocess audio to match input shape [B, 1, 16000]
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, max(0, 16000 - len(y))))
    tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, 16000]
    return tensor

# Inference function
def predict_emotion(audio_path):
    features = preprocess_audio(audio_path)
    with torch.no_grad():
        output = model(features)
        predicted_class = torch.argmax(output, dim=1).item()
    return emotion_labels[predicted_class]

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    emotion = predict_emotion("temp_audio.wav")
    st.success(f"Predicted Emotion: **{emotion}**")
