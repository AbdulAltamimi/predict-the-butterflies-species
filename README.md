# 🦋 Butterfly Species Classification  
<img width="1087" height="204" alt="image" src="https://github.com/user-attachments/assets/2a901e26-0c48-40e4-a524-1d3c920d2abd" />


🧠 EfficientNet-V2 Small + 3-Fold CV + Mixed Precision Training

---

## 📌 Overview

This project implements a complete deep learning pipeline for classifying butterfly species based on images. It uses **EfficientNet-V2 Small** as the backbone, trained with **mixed precision (AMP)** and **3-fold K-Fold cross-validation**.

The final predictions are produced using an **ensemble** of the best models from each fold.

---

## 🚀 Key Features

- ✅ EfficientNet-V2 Small (pretrained on ImageNet)
- ✅ Image size: 224×224
- ✅ Mixed precision training with `torch.cuda.amp`
- ✅ 3-Fold K-Fold Cross-Validation
- ✅ Early Stopping (patience = 3)
- ✅ Ensemble Inference
- ✅ Output: `submission.csv`

---

## 🧪 Training Configuration

| Component     | Value                         |
|---------------|-------------------------------|
| **Model**     | EfficientNet-V2 Small         |
| **Image Size**| 224×224                       |
| **Batch Size**| 32                            |
| **Epochs**    | 10                            |
| **Optimizer** | Adam                          |
| **LR**        | 1e-3                          |
| **Precision** | AMP (Automatic Mixed Precision) |
| **CV Folds**  | 3                             |
| **Early Stop**| Patience = 3                  |

---

## 🖼️ Exploratory Data Analysis (EDA)

- ✅ Visual inspection of sample images
- ✅ Class distribution histogram
- ✅ Total number of unique classes

---

## 📦 DataLoader + Transformations

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```
## 🏗️ Model Architecture
```python
model = torchvision.models.efficientnet_v2_s(pretrained=True)
model.classifier[1] = nn.Linear(in_features, num_classes)
```
## 📈 Performance Summary

| Fold | Best Val F1 Score |
|------|-------------------|
| 1    | 0.8963            |
| 2    | 0.9512            |
| 3    | 0.9790            |
| **Overall OOF Macro-F1** | **0.9434** 🎯 |

> OOF = Out-of-Fold predictions from validation splits.

---

## 📊 Training Loss Curves

Below are the training loss curves recorded for each fold:

<img width="696" height="466" alt="image" src="https://github.com/user-attachments/assets/6570b384-43d6-4820-acbd-5e5ff33053a5" />


---

## 📊 Validation Loss Curves

Below are the validation loss curves recorded for each fold:

<img width="692" height="467" alt="image" src="https://github.com/user-attachments/assets/8afe1085-b86f-4ad1-91d4-35151fa7be92" />


---

## ✅ Final Remarks

This notebook provides a **strong baseline** for butterfly species classification using state-of-the-art techniques:

- ✅ Pretrained EfficientNet-V2 backbone
- ✅ Clean and reproducible data pipeline
- ✅ 3-fold cross-validation with early stopping
- ✅ Ensemble inference with softmax averaging
- ✅ Out-of-Fold Macro F1 Score: **0.9434**

Whether you're participating in a competition or building a real-world classification system, this starter pipeline is a great foundation to build on.


