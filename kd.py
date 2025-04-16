import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from torch.utils.data import Dataset
import os
import torchaudio

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


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha  # Balance between distillation and task loss
        self.temperature = temperature  # Temperature for softening probability distributions
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Task (hard label) loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Distillation (soft label) loss with temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Weighted average of the two losses
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * distillation_loss
        
        return total_loss, hard_loss, distillation_loss

# Hyperparameters
AUDIO_DIR = "/workspace/shripad/Temp/AudioWAV/"
TARGET_LENGTH = 16000
BATCH_SIZE = 64
EPOCHS = 200  # Can be less than teacher model
LEARNING_RATE = 0.001  # Usually higher for student model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_PATH = "crema_cnn_model.pth"  # Path to saved teacher model
TEMPERATURE = 3.0  # Temperature for distillation (higher = softer probability distribution)
ALPHA = 0.7  # Weight for distillation loss (vs regular task loss)

print(f"CUDA is available: {torch.cuda.is_available()}")



# Load dataset
dataset = CremaDataset(audio_dir=AUDIO_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

# Initialize teacher model (same as your original model)
from transformers import Wav2Vec2ForSequenceClassification

teacher_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=6,
    problem_type="single_label_classification"
).to(DEVICE)

# Load teacher model weights
teacher_model.load_state_dict(torch.load(TEACHER_PATH))
teacher_model.eval()  # Set to evaluation mode
print("Teacher model loaded successfully")

# Initialize student model
student_model = LiteCNN(num_classes=6).to(DEVICE)
print(f"Student model is on: {next(student_model.parameters()).device}")

# Print model sizes for comparison
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher_params = count_parameters(teacher_model)
student_params = count_parameters(student_model)
compression_ratio = teacher_params / student_params

print(f"Teacher model parameters: {teacher_params:,}")
print(f"Student model parameters: {student_params:,}")
print(f"Compression ratio: {compression_ratio:.2f}x")

# Optimizer and scheduler
optimizer = Adam(student_model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

# Loss function for knowledge distillation
distillation_criterion = DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)

# Training
start_time = time.time()

for epoch in range(EPOCHS):
    student_model.train()
    train_losses = []
    task_losses = []
    distill_losses = []
    all_preds, all_labels = [], []

    for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        waveforms = waveforms.to(DEVICE)
        labels = labels.to(DEVICE)
        waveforms_s = waveforms.to(DEVICE, non_blocking=True).unsqueeze(1)
        # Forward pass with student model
        student_outputs = student_model(waveforms_s)
        
        # Get teacher predictions (without gradient tracking)
        with torch.no_grad():
            teacher_outputs = teacher_model(waveforms).logits
        
        # Calculate combined loss
        loss, task_loss, distill_loss = distillation_criterion(student_outputs, teacher_outputs, labels)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record metrics
        train_losses.append(loss.item())
        task_losses.append(task_loss.item())
        distill_losses.append(distill_loss.item())
        
        # Track predictions for accuracy calculation
        predictions = torch.argmax(student_outputs, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate training metrics
    train_loss = sum(train_losses) / len(train_losses)
    task_loss_avg = sum(task_losses) / len(task_losses)
    distill_loss_avg = sum(distill_losses) / len(distill_losses)
    train_acc = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f} (Task: {task_loss_avg:.4f}, Distill: {distill_loss_avg:.4f})")
    print(f"Train Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    
    # Validation
    student_model.eval()
    val_losses = []
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            waveforms = waveforms.to(DEVICE)
            waveforms_s = waveforms.to(DEVICE, non_blocking=True).unsqueeze(1)
            labels = labels.to(DEVICE)
            
            # Forward pass with student model
            student_outputs = student_model(waveforms_s)
            
            # Get teacher predictions
            teacher_outputs = teacher_model(waveforms).logits
            
            # Calculate loss
            loss, _, _ = distillation_criterion(student_outputs, teacher_outputs, labels)
            val_losses.append(loss.item())
            
            # Track predictions
            predictions = torch.argmax(student_outputs, dim=1)
            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = sum(val_losses) / len(val_losses)
    val_acc = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    
    # Update learning rate
    scheduler.step(val_loss)

# Calculate and print total training time
end_time = time.time()
print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")



# Save the student model
torch.save(student_model.state_dict(), "crema_student_model_kd.pth")
print("Student model saved successfully!")