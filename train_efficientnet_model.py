import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from model_efficientnet import EfficientNetTextModel
from dataset import GarbageDataset, collate_fn


# ============================================================
# CONFIGURATION
# ============================================================
# Paths to training and validation datasets
train_path = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Train"
val_path   = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Val"

# Training hyperparameters
batch_size = 16
epochs = 5   # short training for fast experimentation

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ============================================================
# DATA AUGMENTATION + PREPROCESSING
# ============================================================
# Resize → random flip → convert to tensor → normalize
# These transforms must match the ones used during evaluation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),  # simple augmentation
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

# Instantiate multimodal EfficientNet model
vocab_size = len(train_dataset.vocab)
num_classes = len(train_dataset.classes)
model = EfficientNetTextModel(vocab_size, num_classes).to(device)


# ============================================================
# LOSS FUNCTION + OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()

# Freeze EfficientNet feature extractor for faster training
for p in model.features.parameters():
    p.requires_grad = False

# Use different learning rates for different parts of the model
optimizer = optim.Adam([
    {"params": model.features.parameters(), "lr": 1e-4},   # pretrained CNN
    {"params": model.image_fc.parameters(), "lr": 1e-3},   # image projection layer
    {"params": model.embedding.parameters(), "lr": 1e-3},  # text embeddings
    {"params": model.classifier.parameters(), "lr": 1e-3}, # final classifier
])


# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, texts, labels in train_loader:
        images = images.to(device)
        texts  = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, texts)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # ---------------- VALIDATION ----------------
    model.eval()
    correct, total = 0, 0

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


# ============================================================
# SAVE FINAL MODEL CHECKPOINT
# ============================================================
torch.save({
    "model_state": model.state_dict(),
    "vocab": train_dataset.vocab,      # save vocabulary for inference
    "classes": train_dataset.classes   # save class names
}, "efficientnet_text_model_final.pth")

print("\nTraining complete. Saved as efficientnet_text_model_final.pth")
