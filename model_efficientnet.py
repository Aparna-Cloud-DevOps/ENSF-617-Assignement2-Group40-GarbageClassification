import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import GarbageDataset, collate_fn


# ============================================================
# MODEL: EfficientNet‑B0 + Text (Multimodal Fusion Model)
# ============================================================
class EfficientNetTextModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # ------------------------------------------------------------
        # IMAGE BRANCH (EfficientNet‑B0 pretrained on ImageNet)
        # ------------------------------------------------------------
        # Load EfficientNet‑B0 with pretrained weights
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Extract only the convolutional feature extractor
        self.features = eff.features

        # Adaptive pooling → ensures output is always 1×1 regardless of input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # EfficientNet‑B0 outputs a 1280‑dimensional feature vector
        # Reduce it to 128 dimensions for fusion with text features
        self.image_fc = nn.Linear(1280, 128)

        # ------------------------------------------------------------
        # TEXT BRANCH (Embedding Layer)
        # ------------------------------------------------------------
        # +1 for padding index; embedding size = 128 to match image feature size
        self.embedding = nn.Embedding(vocab_size + 1, 128, padding_idx=0)

        # ------------------------------------------------------------
        # CLASSIFIER (Fusion Layer)
        # ------------------------------------------------------------
        # Concatenate 128‑dim image features + 128‑dim text features → 256‑dim
        # Final linear layer maps fused features to class logits
        self.classifier = nn.Linear(128 + 128, num_classes)

    def forward(self, image, text):
        # ------------------------------------------------------------
        # IMAGE FORWARD PASS
        # ------------------------------------------------------------
        x = self.features(image)          # CNN feature extraction
        x = self.avgpool(x)               # Global pooling
        x = x.view(x.size(0), -1)         # Flatten to (batch, 1280)
        img_feat = torch.relu(self.image_fc(x))  # Reduce to 128‑dim + ReLU

        # ------------------------------------------------------------
        # TEXT FORWARD PASS
        # ------------------------------------------------------------
        # Convert token IDs → embeddings, then average across sequence length
        text_feat = self.embedding(text).mean(dim=1)

        # ------------------------------------------------------------
        # MULTIMODAL FUSION
        # ------------------------------------------------------------
        # Concatenate image + text features
        combined = torch.cat([img_feat, text_feat], dim=1)

        # Final classification layer
        return self.classifier(combined)
