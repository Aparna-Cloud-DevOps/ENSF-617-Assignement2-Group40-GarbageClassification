import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GarbageDataset, collate_fn

from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn
import torchvision.models as models


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
# Automatically select GPU if available; otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# PATHS
# ============================================================
# train_path → used only to load vocabulary and class names
# test_path  → dataset used for evaluation
# model_path → trained ResNet+Text checkpoint
train_path = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Train"
test_path  = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Test"
model_path = "resnet_text_model_final.pth"


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
# Must match the transforms used during training.
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Standard ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],   # ImageNet mean
        std=[0.229,0.224,0.225]     # ImageNet std
    )
])


# ============================================================
# DATASETS AND DATALOADERS
# ============================================================
# Load training dataset to recover vocabulary and class names
train_dataset = GarbageDataset(train_path, transform=transform)

# Load test dataset using the same vocabulary
test_dataset = GarbageDataset(
    test_path,
    vocab=train_dataset.vocab,
    transform=transform
)

# DataLoader for test set
# - shuffle=False ensures consistent ordering
# - collate_fn handles variable-length text sequences
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    pin_memory=True
)


# ============================================================
# MODEL DEFINITION (ResNet18 + Text Embedding)
# ============================================================
class ResNetTextModel(nn.Module):
    """
    Multimodal model combining:
      • ResNet18 for image feature extraction
      • Simple averaged text embeddings
      • Fully connected classifier for final prediction
    """
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # ---------------- IMAGE BRANCH ----------------
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove final FC layer → output is 512‑dim feature vector
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])

        # Reduce 512‑dim → 128‑dim for fusion
        self.image_fc = nn.Linear(512, 128)

        # ---------------- TEXT BRANCH ----------------
        # Embedding layer converts token IDs → 128‑dim vectors
        self.embedding = nn.Embedding(vocab_size + 1, 128, padding_idx=0)

        # ---------------- CLASSIFIER ----------------
        # Concatenate image (128) + text (128) → 256‑dim
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image, text):
        # ---- IMAGE FORWARD PASS ----
        img_feat = self.image_model(image).view(image.size(0), -1)
        img_feat = torch.relu(self.image_fc(img_feat))

        # ---- TEXT FORWARD PASS ----
        # Average embeddings across sequence length
        text_feat = self.embedding(text).mean(dim=1)

        # ---- MULTIMODAL FUSION ----
        combined = torch.cat((img_feat, text_feat), dim=1)

        # ---- CLASSIFICATION ----
        return self.classifier(combined)


# ============================================================
# LOAD CHECKPOINT
# ============================================================
checkpoint = torch.load(model_path, map_location=device)
vocab   = checkpoint["vocab"]     # vocabulary used during training
classes = checkpoint["classes"]   # class labels

# Initialize model with correct vocab + class count
model = ResNetTextModel(len(vocab), len(classes))

# Load trained weights
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()  # evaluation mode disables dropout/batchnorm updates

print("ResNet+Text Model loaded successfully.")


# ============================================================
# EVALUATION LOOP
# ============================================================
all_preds = []
all_labels = []

# Disable gradient computation for faster inference
with torch.no_grad():
    for images, texts, labels in test_loader:
        # Move batch to GPU/CPU
        images = images.to(device)
        texts  = texts.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, texts)

        # Predicted class = index of max logit
        _, preds = torch.max(outputs, 1)

        # Store predictions + ground truth
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# ============================================================
# METRICS
# ============================================================
# Compute accuracy
accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Print results
print(f"ResNet+Text Model Accuracy: {accuracy:.2f}%")
print("Confusion Matrix:")
print(cm)
