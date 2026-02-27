# Garbage Classification Models Comparison
This repository contains three different models for **garbage classification** using images and text descriptions. The goal is to classify an object into one of the four categories: `Green`, `Blue`, `Black`, or `TTR`.
We compare the performance of:
1. **Image-Only Model** – Simple CNN trained on images.
2. **ResNet + Weights Model** – Pretrained ResNet18 fine-tuned on images.
3. **ResNet + LSTM Model** – Pretrained ResNet18 combined with an LSTM to leverage text descriptions.

## Table of Contents
* [Dataset](#dataset)
* [Models](#models)
* [Setup](#setup)
* [Training](#training)
* [Testing](#testing)
* [Results](#results)
* [Observations](#observations)
* [Future Work](#future-work)

## Dataset
* **Training Dataset**: `/CVPR_2024_dataset_Train` (~11,629 images)
* **Validation Dataset**: `/CVPR_2024_dataset_Val` (~1,800 images)
* **Test Dataset**: `/CVPR_2024_dataset_Test` (~size depends on your dataset)
* Each sample contains:
  * **Image** of garbage object
  * **Text description** (used only for multimodal model)
  * **Label**: `Green`, `Blue`, `Black`, or `TTR`
    
## Models
### 1️⃣ Image-Only Model
* **Architecture**: Simple CNN with 3 convolutional layers + adaptive pooling + fully connected classifier.
* **Input**: Images only.
* **Output**: Class probabilities for 4 categories.
* **Advantages**: Fast to train, minimal GPU memory.
* **Limitations**: Cannot leverage text information; lower accuracy if visual features are ambiguous.

### 2️⃣ ResNet + Weights Model

* **Architecture**: Pretrained ResNet18 (ImageNet weights)
* **Input**: Images only
* **Modifications**: Replace final fully connected layer to match 4 classes.
* **Advantages**: Strong feature extractor; can capture fine-grained image details.
* **Limitations**: Still ignores text; larger model size and longer training time.

### 3️⃣ ResNet + LSTM Model

* **Architecture**: Pretrained ResNet18 + LSTM for text embeddings.
* **Input**: Images + Text
* **Fusion**: Concatenate image features (from ResNet) and text features (from LSTM) before classifier.
* **Advantages**: Multimodal; can leverage complementary information from text.
* **Limitations**: Slower training; more memory-intensive.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/garbage-classification.git
cd garbage-classification
```
2. Create environment:

```bash
conda create -n garbage_env python=3.10
conda activate garbage_env
pip install -r requirements.txt
```
**Requirements**:

* torch >= 2.0
* torchvision >= 0.17
* PIL / Pillow
* numpy
  
3. Place datasets in folders:

```text
garbage_data/
 ├── CVPR_2024_dataset_Train/
 ├── CVPR_2024_dataset_Val/
 └── CVPR_2024_dataset_Test/
```
## Training

Each model has a separate training file:

| Model            | Training Script           |
| ---------------- | ------------------------- |
| Image-Only       | `train_image_only.py`     |
| ResNet + Weights | `train_resnet_weights.py` |
| ResNet + LSTM    | `train_resnet_lstm.py`    |

```bash
# Example: Train Image-Only Model
python train_image_only.py
```
**Notes**:

* Batch size, epochs, and learning rate are configurable at the top of each script.
* Checkpoints are saved in `checkpoints/` per epoch.
* Final models are saved as `.pth` files.

## Testing

Each model has a corresponding test script:

| Model            | Test Script              |
| ---------------- | ------------------------ |
| Image-Only       | `test_image_only.py`     |
| ResNet + Weights | `test_resnet_weights.py` |
| ResNet + LSTM    | `test_resnet_lstm.py`    |

```bash
# Example: Test ResNet + LSTM
python test_resnet_lstm.py

**Test Output Includes**:

* Test accuracy (%)
* Confusion matrix
* Optionally: per-class precision/recall

## Results (Example)

| Model            | Accuracy (%) |
| ---------------- | ------------ |
| Image-Only       | 21.79        |
| ResNet + Weights | 75.32        |
| ResNet + LSTM    | 83.50        |

**Confusion matrices** and more detailed metrics can be found in the respective test scripts.

## Observations

* Image-only CNN performs poorly for this dataset due to ambiguous images.
* Pretrained ResNet18 significantly improves accuracy, even without text.
* ResNet + LSTM (multimodal) gives the best performance by leveraging both visual and textual information.
* Training for only a few epochs may suffice for ResNet features, while image-only CNN benefits from longer training.

## Future Work

* Experiment with **larger pretrained networks** (ResNet34, ResNet50).
* Use **data augmentation** to improve generalization.
* Try **attention mechanisms** for better text-image fusion.
* Explore **transformer-based models** for multimodal inputs.

This README will give a **clear overview for someone cloning your repo** and allow them to quickly understand your models, dataset, training/testing procedure, and comparative results.


Do you want me to do that next?
# ENSF-617-Assignement2-Group40-GarbageClassification
