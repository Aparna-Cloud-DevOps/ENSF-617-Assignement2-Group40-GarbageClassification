import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ============================================================
# MULTIMODAL MODEL: IMAGE (ResNet18) + TEXT (LSTM)
# ============================================================
class ResNetLSTMModel(nn.Module):
    """
    Multimodal architecture combining:
      • ResNet18 for image feature extraction
      • LSTM for text sequence modeling
      • Fully connected classifier for final prediction
    """

    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # ------------------------------------------------------------
        # IMAGE BRANCH — Pretrained ResNet18
        # ------------------------------------------------------------
        # Load ResNet18 pretrained on ImageNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final classification layer (fc)
        # Output of this truncated network is a 512‑dim feature vector
        self.image