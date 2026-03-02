import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# Text cleaning function
# Applied before tokenization for consistent vocabulary
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower()                               # lowercase
    text = re.sub(r"[^a-z0-9 ]", " ", text)           # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()          # collapse spaces
    return text

# ---------------------------------------------------------
# Custom collate function
# Pads variable-length text sequences in a batch
# ---------------------------------------------------------
def collate_fn(batch):
    images, texts, labels = zip(*batch)

    texts = [t.tolist() if isinstance(t, torch.Tensor) else t for t in texts]
    max_len = max(len(t) for t in texts)
    padded_texts = [t + [0] * (max_len - len(t)) for t in texts]

    images = torch.stack(images)
    texts = torch.tensor(padded_texts)
    labels = torch.tensor(labels)

    return images, texts, labels

# ---------------------------------------------------------
# GarbageDataset
# Loads images and extracts text from filenames.
# ---------------------------------------------------------
class GarbageDataset(Dataset):

    def __init__(self, root_dir, vocab=None, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Folder names = class labels
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Build vocabulary only for training
        if vocab is None:
            self.vocab = {}
            self.build_vocab()
        else:
            self.vocab = vocab

        # Collect image paths, cleaned text, and labels
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):

                    img_path = os.path.join(class_path, file)

                    # Extract text from filename
                    text = file.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                    text = text.replace("_", " ")

                    # CLEAN TEXT HERE
                    text = clean_text(text)

                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, text, label))


    # -----------------------------------------------------
    # Build vocabulary from cleaned training filenames
    # -----------------------------------------------------
    def build_vocab(self):
        index = 1  # 0 is reserved for padding

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)

            for file in os.listdir(class_path):
                text = file.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                text = text.replace("_", " ")

                # CLEAN TEXT BEFORE TOKENIZATION
                text = clean_text(text)

                words = text.split()

                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = index
                        index += 1


    # -----------------------------------------------------
    # Convert cleaned text string into tensor of word indices
    # -----------------------------------------------------
    def text_to_tensor(self, text):
        words = text.split()
        indices = [self.vocab.get(word, 0) for word in words]  # unknown → 0
        return torch.tensor(indices, dtype=torch.long)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, text, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # CLEAN TEXT AGAIN (safety for test/eval)
        text = clean_text(text)

        # Convert to tensor
        text_tensor = self.text_to_tensor(text)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, text_tensor, label_tensor
