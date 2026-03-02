import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GarbageDataset, collate_fn
from model_image_only import ImageOnlyModel  # Image-only CNN model definition

from sklearn.metrics import confusion_matrix
import numpy as np


# ============================================================
# DEVICE CONFIGURATION
# ============================================================
# Automatically select GPU if available; otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# PATHS
# ============================================================
# Dataset directories and trained model checkpoint
train_path = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Train"
test_path  = "/home/aparna.ayyalasomayaj/garbage_data/CVPR_2024_dataset_Test"
model_path = "garbage_model_image_only.pth"  # trained image-only model checkpoint


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
# Transformations must match those used during training.
# Resize → convert to tensor → normalize using ImageNet stats.
transform = transforms.Compose([
    transforms.Resize((128,128)),  # small resolution suitable for simple CNN
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],   # ImageNet mean
        std=[0.229,0.224,0.225]     # ImageNet std
    )
])


# ============================================================
# DATASETS AND DATALOADERS
# ============================================================
# Load training dataset only to recover class labels (stored in checkpoint)
train_dataset = GarbageDataset(train_path, transform=transform)

# Load test dataset for evaluation
test_dataset = GarbageDataset(test_path, transform=transform)

# DataLoader for test set
# - shuffle=False ensures consistent ordering
# - collate_fn handles text fields (ignored for image-only model)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    pin_memory=True
)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================
# Load checkpoint containing model weights + class names
checkpoint = torch.load(model_path, map_location=device)
classes = checkpoint["classes"]  # list of class labels

# Initialize model with correct number of output classes
model = ImageOnlyModel(len(classes))

# Load trained weights
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()  # evaluation mode disables dropout/batchnorm updates

print("Model loaded successfully.")


# ============================================================
# EVALUATION ON TEST DATA
# ============================================================
all_preds = []
all_labels = []

# Disable gradient computation for faster inference
with torch.no_grad():
    for images, texts, labels in test_loader:
        # Move images + labels to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass through the image-only CNN
        outputs = model(images)

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
print(f"Image-only Model Accuracy: {accuracy:.2f}%")
print("Confusion Matrix:")
print(cm)
