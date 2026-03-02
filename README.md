# **Garbage Classification Using Image, Text, and Multimodal Deep Learning Models**
# **Submitted By : **
Ayyalasomayajula, Aparna - 30303831
Radadiya, Mansi Mukeshbhai - 30303863
This repository contains a complete experimental pipeline for garbage classification using **images**, **text descriptions**, and **multimodal fusion architectures**. The goal is to classify each sample into one of four categories:

- **Green**
- **Blue**
- **Black**
- **TTR**

The project evaluates multiple model families—image‑only CNNs, pretrained ResNet and EfficientNet backbones, and multimodal architectures combining image and text—to understand how different modalities contribute to classification accuracy.

## **1. Dataset**

Each sample in the dataset contains:

- An **image** of a garbage item  
- A **text description** (used in multimodal models)  
- A **label**: Green, Blue, Black, or TTR  

Directory structure:

```
garbage_data/
 ├── CVPR_2024_dataset_Train/
 ├── CVPR_2024_dataset_Val/
 └── CVPR_2024_dataset_Test/

Dataset variants used:

- **Image‑only dataset** (for CNN, ResNet, EfficientNet image models)
- **ResNet multimodal dataset** (image + tokenized text)
- **EfficientNet multimodal dataset** (image + tokenized text)

## **2. Experiment Setup**

### **Why These Models Were Chosen**

- **Image‑Only CNN (baseline)**  
  Establishes how well the task can be solved using visual information alone.

- **ResNet‑Text Multimodal Model**  
  Combines ResNet image embeddings with text embeddings to test whether textual descriptions improve classification.

- **Multimodal LSTM Model**  
  Uses ResNet image features + an LSTM encoder for text to evaluate whether sequential modeling improves multimodal fusion.

- **EfficientNet‑Text Multimodal Model**  
  Tests whether a more efficient image backbone paired with text improves performance compared to ResNet‑based multimodal models.

### **What the Experiments Aim to Achieve**

- How much does **text** improve classification compared to image‑only models?  
- Which **multimodal fusion strategy** performs best?  
- What types of **errors** do different models make?  
- How do **pretrained backbones** compare to simpler CNNs?

## **3. Models Included**

### **1️⃣ Image‑Only Model**
- Simple CNN or MobileNet/EfficientNet backbone  
- Input: Images  
- Output: 4‑class prediction  
- Purpose: Baseline performance  

### **2️⃣ ResNet‑Text Multimodal Model**
- Pretrained ResNet18  
- Text embeddings averaged and fused with image features  
- Purpose: Strong multimodal baseline  

### **3️⃣ Multimodal LSTM Model**
- ResNet image encoder  
- LSTM text encoder  
- Fusion → classifier  
- Purpose: Sequential text modeling  

### **4️⃣ EfficientNet‑Text Multimodal Model**
- EfficientNet‑B0 image encoder  
- Text embedding + fusion  
- Purpose: Efficient multimodal architecture  

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
- torch ≥ 2.0  
- torchvision ≥ 0.17  
- numpy  
- Pillow  
- matplotlib  

## **5. Training**

Each model has its own training script:

| Model | Training Script |
|-------|-----------------|
| Image‑Only | `train_image_only.py` |
| ResNet‑Text | `train_resnet_text.py` |
| Multimodal LSTM | `train_resnet_lstm.py` |
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

- **Wrong Predictions Per Class**  
  ```python
  show_wrong_predictions_per_class(preds, labels, dataset, class_names)
  ```

- **Correct Predictions Per Class**  
  ```python
  show_correct_predictions_per_class(preds, labels, dataset, class_names)
  ```

These visualizations help identify:

- Systematic confusion patterns  
- Which classes benefit most from multimodal fusion  
- Whether errors arise from image ambiguity or text ambiguity  


## **8. Running Training and Testing on the GPU Cluster**

All training and testing was executed on the university GPU cluster using SLURM.

### **Submitting a GPU Job**

```
sbatch gpu_job.slurm
```

Each model was trained and tested by modifying the Python script name inside the SLURM file.

### **Scripts Run via SLURM**

| Model | Training | Testing |
|-------|----------|---------|
| Image‑Only | `train_image_only.py` | `test_image_only.py` |
| ResNet‑Text | `train_resnet_text.py` | `test_resnet_text.py` |
| ResNet‑LSTM | `train_resnet_lstm.py` | `test_resnet_lstm.py` |
| EfficientNet‑Text | `train_efficientnet_text.py` | `test_efficientnet.py` |

---

## **9. Running Jupyter Notebook on the GPU Cluster**

Jupyter Notebook was used for:

- Visualizing predictions  
- Inspecting confusion matrices  
- Debugging dataset issues  
- Running evaluation interactively  

### **Step 1 — Start Jupyter on the cluster**

```
module load cuda/12.6.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpuenv

jupyter notebook --no-browser --port=8888
```

### **Step 2 — Create SSH tunnel from local machine**

```
ssh -N -L 8888:localhost:8888 <username>@<cluster-address>
```

### **Step 3 — Open in browser**

Navigate to:

```
http://localhost:8888
```

Paste the token printed in the cluster terminal.

## **10. Results**

All results appear in the notebook:

- Classification reports  
- Confusion matrices  
- Correct/incorrect prediction visualizations  

Results are grouped by model:

- Image‑Only  
- ResNet‑Text  
- Multimodal LSTM  
- EfficientNet‑Text  

## **11. Observations**

- Image‑only models struggle with ambiguous or visually similar items.  
- Pretrained backbones significantly improve accuracy.  
- Multimodal models outperform image‑only models by leveraging text.  
- LSTM‑based text modeling helps with longer descriptions.  
- EfficientNet‑Text provides strong performance with lower computational cost.  

## **12. Future Work**

- Transformer‑based multimodal fusion (ViLT, CLIP‑style)  
- Attention mechanisms for better text‑image alignment  
- Hyperparameter tuning for fusion layers  
- Data augmentation and contrastive learning  
