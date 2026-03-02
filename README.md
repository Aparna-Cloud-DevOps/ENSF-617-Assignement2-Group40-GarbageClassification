# **README.txt**

## **Garbage Classification Using Image, Text, and Multimodal Deep Learning Models**

This repository contains a complete experimental pipeline for garbage classification using **images**, **text descriptions**, and **multimodal fusion architectures**. The goal is to classify each sample into one of four categories:

- **Green**
- **Blue**
- **Black**
- **TTR**

The project evaluates multiple model families—image‑only CNNs, pretrained ResNet and EfficientNet backbones, and multimodal architectures combining image and text—to understand how different modalities contribute to classification accuracy.

---

## **1. Dataset**

The dataset consists of images paired with short text descriptions. Each sample includes:

- An **image** of a garbage item  
- A **text description** (used in multimodal models)  
- A **label**: Green, Blue, Black, or TTR  

Dataset structure:

```
garbage_data/
 ├── CVPR_2024_dataset_Train/
 ├── CVPR_2024_dataset_Val/
 └── CVPR_2024_dataset_Test/
```

Three dataset variants are used in the code:

- **Image‑only dataset** (for MobileNet/EfficientNet/ResNet image models)
- **ResNet multimodal dataset** (image + tokenized text)
- **EfficientNet multimodal dataset** (image + tokenized text)

---

## **2. Experiment Setup**

### **Why These Models Were Chosen**

The models were selected to explore how different modalities and architectures affect classification performance:

- **Image‑Only CNN (baseline)**  
  Establishes how well the task can be solved using visual information alone.

- **ResNet‑Text Multimodal Model**  
  Combines ResNet image embeddings with text embeddings to test whether textual descriptions improve classification.

- **Multimodal LSTM Model**  
  Uses ResNet image features + an LSTM encoder for text to evaluate whether sequential modeling of text improves multimodal fusion.

- **EfficientNet‑Text Multimodal Model**  
  Tests whether a more efficient image backbone paired with text improves performance compared to ResNet‑based multimodal models.

### **What the Experiments Aim to Achieve**

The experiments are designed to answer:

1. How much does **text** improve classification compared to image‑only models?  
2. Which multimodal fusion strategy performs best?  
3. What types of errors do different models make, and are they systematic?  
4. How do pretrained backbones (ResNet, EfficientNet) compare to simpler CNNs?

---

## **3. Models Included**

### **1️⃣ Image‑Only Model**
- Simple CNN or MobileNet/EfficientNet backbone  
- Input: Images  
- Output: 4‑class prediction  
- Purpose: Baseline performance  

### **2️⃣ ResNet‑Text Multimodal Model**
- Pretrained ResNet18 for image features  
- Text embedded + fused with image features  
- Purpose: Evaluate multimodal fusion with a strong image backbone  

### **3️⃣ Multimodal LSTM Model**
- ResNet image encoder  
- LSTM text encoder  
- Fusion: concatenation → classifier  
- Purpose: Test sequential text modeling  

### **4️⃣ EfficientNet‑Text Multimodal Model**
- EfficientNet image encoder  
- Text embedding + fusion  
- Purpose: Efficient multimodal architecture  

---

## **4. Setup Instructions**

### **Clone the repository**
```
git clone https://github.com/<your-username>/ENSF-617-Assignement2-Group40-GarbageClassification.git
cd ENSF-617-Assignement2-Group40-GarbageClassification
```

### **Create environment**
```
conda create -n garbage_env python=3.10
conda activate garbage_env
pip install -r requirements.txt
```

### **Requirements**
- torch >= 2.0  
- torchvision >= 0.17  
- numpy  
- Pillow  
- matplotlib  

---

## **5. Training**

Each model has its own training script:

| Model | Training Script |
|-------|-----------------|
| Image‑Only | `train_image_only.py` |
| ResNet‑Text | `train_resnet_text.py` |
| Multimodal LSTM | `train_multimodal_lstm.py` |
| EfficientNet‑Text | `train_efficientnet_text.py` |

Example:
```
python train_resnet_text.py
```

Checkpoints are saved automatically.

---

## **6. Testing and Evaluation**

All models are evaluated using:

```python
evaluate_model(model, dataloader, model_name)
```

This function:

- Loads the correct checkpoint  
- Runs inference  
- Computes accuracy, precision, recall, F1‑score  
- Prints a classification report  
- Displays a confusion matrix  
- Returns predictions + labels  

---

## **7. Visualization of Predictions**

Two helper functions provide qualitative insights:

### **Wrong Predictions Per Class**
```python
show_wrong_predictions_per_class(preds, labels, dataset, class_names)
```
Displays misclassified images grouped by true class.

### **Correct Predictions Per Class**
```python
show_correct_predictions_per_class(preds, labels, dataset, class_names)
```
Displays correctly classified images grouped by class.

These visualizations help identify:

- Systematic confusion patterns  
- Which classes benefit most from multimodal fusion  
- Whether errors arise from image ambiguity or text ambiguity  

---

## **8. Results**

All results appear **directly in the notebook**, including:

- Classification reports  
- Confusion matrices  
- Correct prediction visualizations  
- Wrong prediction visualizations  

Results are grouped by model:

- **Image‑Only Model Results**  
- **ResNet‑Text Model Results**  
- **Multimodal LSTM Results**  
- **EfficientNet‑Text Results**  

Each section includes both quantitative and qualitative outputs.

---

## **9. Observations**

- Image‑only models struggle with ambiguous or visually similar items.  
- Pretrained backbones (ResNet, EfficientNet) significantly improve accuracy.  
- Multimodal models outperform image‑only models by leveraging text.  
- LSTM‑based text modeling improves performance for longer or more descriptive text.  
- EfficientNet‑Text provides strong performance with lower computational cost.  

---

## **10. Future Work**

- Explore transformer‑based multimodal fusion (e.g., ViLT, CLIP‑style models).  
- Add attention mechanisms for better text‑image alignment.  
- Perform hyperparameter tuning for fusion layers.  
- Add data augmentation and contrastive learning.  

If you'd like, I can also generate a **final comparison table** summarizing all model metrics for your report.
