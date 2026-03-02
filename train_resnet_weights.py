import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from model import ResNetTextModel
from dataset import GarbageDataset, collate_fn


# ============================================================
# TRAINING CONFIGURATION
# ============================================================
# Paths to training and validation datasets
train_path = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Train"
val_path   = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Val"

# Hyperparameters
batch_size = 16
epochs = 5
learning_rate = 1e-3

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
# Resize → convert to tensor → normalize
# These transforms must match the ones used during evaluation
transform = transforms.Compose([
    transforms.Resize((224,224)),  # ResNet18 expects 224×224 images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])


# ============================================================
# DATASETS AND DATALOADERS
# ============================================================
# Training dataset builds vocabulary from text descriptions
train_dataset = GarbageDataset(train_path, transform=transform)

# Validation dataset must reuse the same vocabulary
val_dataset = GarbageDataset(
    val_path,
    vocab=train_dataset.vocab,
    transform=transform
)

# DataLoaders for batching
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=1,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    pin_memory=True
)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))


# ============================================================
# MODEL INITIALIZATION (ResNet18 + Text Embedding)
# ============================================================
vocab_size = len(train_dataset.vocab)
num_classes = len(train_dataset.classes)

# Instantiate multimodal model
model = ResNetTextModel(vocab_size, num_classes).to(device)


# ============================================================
# LOSS FUNCTION + OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    # ---------------- TRAINING ----------------
    for images, texts, labels in train_loader:
        images = images.to(device)
        texts  = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass through multimodal model
        outputs = model(images, texts)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images = images.to(device)
            texts  = texts.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # ---------------- SAVE CHECKPOINT ----------------
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/resnet_text_epoch_{epoch+1}.pth"

    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, ckpt_path)

    print(f"Checkpoint saved: {ckpt_path}\n")


# ============================================================
# SAVE FINAL MODEL
# ============================================================
torch.save({
    "model_state": model.state_dict(),
    "vocab": train_dataset.vocab,      # save vocabulary for inference
    "classes": train_dataset.classes   # save class names
}, "resnet_text_model_final.pth")

print("Training completed and model saved as resnet_text_model_final.pth")