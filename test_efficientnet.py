import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import GarbageDataset, collate_fn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


# ============================================================
# MODEL: EfficientNet‑B0 + Text (same architecture used in training)
# ============================================================
class EfficientNetTextModel(nn.Module):
    """
    Multimodal model combining:
    - EfficientNet‑B0 for image feature extraction
    - Embedding layer for text feature extraction
    - Concatenation-based fusion for classification
    """
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # ---------------- IMAGE BRANCH ----------------
        # Load pretrained EfficientNet‑B0
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Extract convolutional feature extractor
        self.features = eff.features

        # Global average pooling → output shape (batch, 1280, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Reduce 1280‑dim EfficientNet features → 128‑dim
        self.image_fc = nn.Linear(1280, 128)

        # ---------------- TEXT BRANCH ----------------
        # Embedding layer converts token IDs → 128‑dim vectors
        self.embedding = nn.Embedding(vocab_size + 1, 128, padding_idx=0)

        # ---------------- CLASSIFIER ----------------
        # Concatenate image (128) + text (128) → 256‑dim
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image, text):
        # ---- IMAGE FORWARD PASS ----
        x = self.features(image)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)              # flatten to (batch, 1280)
        img_feat = torch.relu(self.image_fc(x))  # reduce to 128‑dim

        # ---- TEXT FORWARD PASS ----
        # Average embeddings across sequence length
        text_feat = self.embedding(text).mean(dim=1)

        # ---- MULTIMODAL FUSION ----
        combined = torch.cat([img_feat, text_feat], dim=1)

        return self.classifier(combined)


# ============================================================
# CONFIGURATION
# ============================================================
test_path = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Test"
checkpoint_path = "efficientnet_text_model_final.pth"

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ============================================================
# IMAGE TRANSFORMS (must match training/validation transforms)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),  # EfficientNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CHECKPOINT (model weights + vocab + class names)
# ============================================================
ckpt = torch.load(checkpoint_path, map_location=device)
vocab = ckpt["vocab"]
class_names = ckpt["classes"]
num_classes = len(class_names)

print("Loaded checkpoint:", checkpoint_path)
print("Classes:", class_names)


# ============================================================
# LOAD TEST DATASET
# ============================================================
test_dataset = GarbageDataset(
    test_path,
    vocab=vocab,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1
)

print("Test size:", len(test_dataset))


# ============================================================
# INITIALIZE MODEL + LOAD TRAINED WEIGHTS
# ============================================================
model = EfficientNetTextModel(
    vocab_size=len(vocab),
    num_classes=num_classes
)

model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

print("Model loaded and ready.")


# ============================================================
# RUN INFERENCE ON TEST SET
# ============================================================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, texts, labels in test_loader:
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, texts)

        # Predicted class = argmax(logits)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# ============================================================
# METRICS AND REPORTING
# ============================================================
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


# ============================================================
# SAVE PREDICTIONS TO CSV
# ============================================================
df = pd.DataFrame({
    "true": [class_names[i] for i in all_labels],
    "pred": [class_names[i] for i in all_preds]
})

df.to_csv("efficientnet_test_predictions.csv", index=False)
print("\nSaved predictions to efficientnet_test_predictions.csv")
