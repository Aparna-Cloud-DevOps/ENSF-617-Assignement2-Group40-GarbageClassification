import torch
import torch.nn as nn
import torchvision.models as models

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ------------------------------------------------------------
        # IMAGE FEATURE EXTRACTOR (Simple CNN)
        # ------------------------------------------------------------
        # This CNN acts as a lightweight baseline model.
        # It progressively extracts spatial features using:
        #   - Convolution layers
        #   - ReLU activations
        #   - MaxPooling for downsampling
        # The final AdaptiveAvgPool2d ensures a fixed 1×1 output
        # regardless of input image size.
        self.features = nn.Sequential(
            # First conv block: extract low-level edges/textures
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2

            # Second conv block: deeper spatial features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2

            # Third conv block: high-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Global pooling → output shape becomes (batch, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1,1))
        )

        # ------------------------------------------------------------
        # CLASSIFIER
        # ------------------------------------------------------------
        # After pooling, the feature map is flattened to 128 units.
        # A single fully connected layer maps features → class logits.
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass image through CNN feature extractor
        x = self.features(x)

        # Flatten from (batch, 128, 1, 1) → (batch, 128)
        x = x.view(x.size(0), -1)

        # Final classification layer
        x = self.classifier(x)
        return x
