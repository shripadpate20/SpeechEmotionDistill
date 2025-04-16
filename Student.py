import os
import torchaudio
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn.functional as nn
import torch.nn as nn  
import time
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# CremaDataset to load raw waveforms
class CremaDataset(Dataset):
    def __init__(self, audio_dir, target_length=16000, augment=True):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.augment = augment
        self.target_length = target_length  # Target length in samples (16kHz sample rate)
        
        self.emotion_to_idx = {
            "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "SAD": 4, "NEU": 5
        }

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filename = self.audio_files[idx]
        emotion_code = filename.split("_")[2]
        label = self.emotion_to_idx[emotion_code]

        audio_path = os.path.join(self.audio_dir, filename)
        
        # Load raw waveform and resample to 16kHz
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Truncate/pad to fixed length
        if waveform.shape[1] < self.target_length:
            waveform = F.pad(waveform, (0, self.target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_length]
        
        return waveform.squeeze(0), label  # Shape: [16000]





class LiteCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv with larger stride
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 8000]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B, 32, 4000]
            
            # Depthwise separable convolution
            nn.Conv1d(32, 64, kernel_size=3, groups=32, padding=1),  # Depthwise
            nn.Conv1d(64, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [B, 64, 2000]
            
            # Bottleneck layer
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)  # [B, 128, 64]
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128*64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    



# Hyperparameters
AUDIO_DIR = "/workspace/shripad/Temp/AudioWAV/" 
TARGET_LENGTH = 16000
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")

# Dataset and Loaders
dataset = CremaDataset(audio_dir=AUDIO_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,pin_memory=True,num_workers=4,shuffle=True)
val_loader = DataLoader(val_dataset,pin_memory=True,num_workers=4 ,batch_size=BATCH_SIZE)

# Model, Loss, Optimizer, Scheduler
model = LiteCNN().to(DEVICE)
print(f"Model is on: {next(model.parameters()).device}")
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

# Training
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    all_preds, all_labels = [], []

    for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
       
        labels = labels.to(DEVICE, non_blocking=True)
        waveforms = waveforms.to(DEVICE, non_blocking=True).unsqueeze(1)  # Adds channel dimension

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

    # Validation
    model.eval()
    val_losses = []
    val_preds, val_labels = [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            waveforms = waveforms.to(DEVICE, non_blocking=True).unsqueeze(1)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = sum(val_losses) / len(val_losses)
    val_acc = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, "
          f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    # Step scheduler
    scheduler.step(train_loss)

end_time = time.time()
print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")

# Optional: Save the model
torch.save(model.state_dict(), "crema_cnn_model_student.pth")
