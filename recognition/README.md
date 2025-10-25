# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion


# ADNI MRI Classification with ConvNeXt

This project implements a **ConvNeXt-Tiny model** to classify Alzheimer's Disease (AD) versus Normal Control (NC) from MRI slice images in the **ADNI dataset**. It provides a complete training and prediction pipeline in PyTorch, including data augmentation, evaluation metrics, and model checkpointing.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Requirements](#requirements)  
- [Dataset](#dataset)  
- [Training](#training)  
- [Prediction](#prediction)  
- [Metrics](#metrics)  
- [Notes](#notes)  

---

## Project Structure

├── dataset.py # Data loader, augmentation, mixup function
├── modules.py # Model definitions (ConvNeXtTiny and ADNIConvNext)
├── train.py # Training script
├── predict.py # Prediction / evaluation script
└── README.md

- `dataset.py`  
  - Implements `ADNIDataset` and `get_dataloaders()`.  
  - Supports train/validation/test splits and mixup augmentation.

- `modules.py`  
  - Implements `ConvNeXtTiny` and custom wrapper `ADNIConvNext`.  
  - Includes dropout control.

- `train.py`  
  - Trains the model on the ADNI dataset.  
  - Supports early stopping, learning rate scheduling, and model checkpointing.  
  - Generates training curves for loss and accuracy.

- `predict.py`  
  - Loads a saved checkpoint and evaluates on a folder of test images.  
  - Supports single-image prediction.  
  - Returns accuracy, sensitivity (recall), specificity, and AUC.  

---

## Requirements

Install the necessary dependencies via pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm pillow
```

---

## Dataset

Expected file structure

dataset/ADNI/AD_NC/
├── train/
│   ├── AD/
│   └── NC/
├── test/
│   ├── AD/
│   └── NC/

- Images must be in .jpg, .jpeg, or .png format.
- Labels: AD = 1, NC = 0.

---

## Training

Run the training script `train.py`

- Uses ConvNeXt-Tiny from scratch.
- Training configuration is defined in train.py (batch size, learning rate, dropout, etc.).
- Supports Mixup augmentation for generalisation (controlled by mixup_alpha).
- Early stopping is implemented with patience of 15 epochs.
- Model checkpoints are saved automatically in checkpoints/.

Training outputs:
- training_curve_<timestamp>.png — Visualizes loss and accuracy over epochs.
- Model checkpoint .pth file for later evaluation/prediction.

---

## Prediction

Run the prediction/evaluation script `predict.py`

- Loads a saved checkpoint (set checkpoint_path in predict.py).
- Evaluates all images in a test folder (dataset/ADNI/AD_NC/test).
- Predicts a single image as well.

Example output:

Test Folder Evaluation:
Accuracy: 0.9123
Sensitivity (Recall): 0.9300
Specificity: 0.8950
AUC: 0.9575

Single Image Prediction:
Predicted Class: AD with confidence 0.9843

---

## Metrics

The evaluation includes:
- Accuracy — overall classification correctness
- Sensitivity — correctly identified AD cases
- Specificity — correctly identified NC cases
- AUC — area under the ROC curve

---

## Notes

- Images are resized to 224x224 for compatibility with ConvNeXt.
- Training uses label smoothing and AdamW optimizer with cosine annealing.
- Dropout can be adjusted via model.update_dropout_rate() in train.py.
- Mixup augmentation is optional (mixup_alpha in train.py).

