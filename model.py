import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------
# GarbageModel
# Multimodal model that combines:
# 1) Image features (ResNet18)
# 2) Text features (word embeddings)
# ---------------------------------------------------------
class ResNetTextModel(nn.Module):
    """
    Multimodal model combining:
    - ResNet18 for image feature extraction
    - Text embedding with average pooling for text feature
    - Concatenation of image + text features followed by classification
    """
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # -------- IMAGE BRANCH --------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # pretrained ResNet18
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])   # remove final fc layer
        self.image_fc = nn.Linear(512, 128)                               # map 512-d image feature to 128-d

        # -------- TEXT BRANCH --------
        # Embedding layer converts vocab indices to 128-d vectors
        self.embedding = nn.Embedding(vocab_size + 1, 128, padding_idx=0)

        # -------- CLASSIFIER --------
        # Concatenate 128-d image + 128-d text = 256-d features -> output num_classes
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image, text):
        """
        Forward pass:
        - Extract image features using ResNet18
        - Extract text features using embedding + mean pooling
        - Concatenate image + text features
        - Classify using final linear layer
        """
        # Image features
        img_feat = self.image_model(image).view(image.size(0), -1)  # flatten
        img_feat = torch.relu(self.image_fc(img_feat))               # fully connected + ReLU

        # Text features: average embeddings over sequence length
        text_feat = self.embedding(text).mean(dim=1)

        # Concatenate image + text features
        combined = torch.cat((img_feat, text_feat), dim=1)

        # Output logits for classification
        output = self.classifier(combined)
        return output
