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
- [Outputs](#outputs) 

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
  - Implements `ConvNeXtTiny`.  
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
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 numpy==2.1.2 matplotlib==3.10.6 scikit-learn==1.7.2 tqdm==4.67.1 pillow==11.0.0
```

---

## Dataset

Expected file structure

/home/groups/comp3710/ADNI/AD_NC/
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

Run the training script `train.py` from inside the `/adni_convnext_46936125` directory:

- Uses ConvNeXt-Tiny from scratch.
- Training configuration is defined in train.py (batch size, learning rate, dropout, etc.).
- Supports Mixup augmentation for generalisation (controlled by mixup_alpha).
- Early stopping WAS implemented based on no improvement epochs but has been removed based on testing, checkpointing (saving the best model determined by validation accuracy) is now being used alone.
- Model checkpoints are saved automatically in checkpoints/.

Training outputs:
- training_curve.png — Visualizes loss and accuracy over epochs.
- confusion_matrix.png — Shows the subject-level classification results (True vs Predicted) to reveal misclassifications.
- roc_curve.png — Displays the Receiver Operating Characteristic curve, showing model performance across thresholds.
- Model checkpoint .pth file for later evaluation/prediction.

---

## Prediction

Run the prediction/evaluation script `predict.py` from inside the `/adni_convnext_46936125` directory:

- Loads a saved checkpoint (set checkpoint_path in predict.py).
- Evaluates all images in a test folder (/home/groups/comp3710/ADNI/AD_NC/test).
- Predicts a single image as well.

---

## Metrics

The evaluation includes:
- AUC — area under the ROC curve

---

## Notes

- Images are resized to 224x224 for compatibility with ConvNeXt.
- Training uses label smoothing and AdamW optimizer with cosine annealing.
- Mixup augmentation is optional (mixup_alpha in train.py).

---

## Outputs

Detailed are the models performance on the provided test dataset, it is noted that the 0.8 accuracy level was not achieved (sadness).

Test Folder Evaluation (Subject-Level):
Overall accuracy: 0.7844
Correct AD classification accuracy: 0.7354
Correct NC classification accuracy: 0.8326
AUC: 0.8616

Single Image Prediction: /home/groups/comp3710/ADNI/AD_NC/test/AD/388206_78.jpeg
Predicted Class: AD with confidence 0.9684

Outputs ROC curve, Confusion matrix and training curve to 'checkpoints'

![Training Curve](checkpoints/training_curve.png)
![Confusion Matrix](checkpoints/confusion_matrix.png)
![ROC Curve](checkpoints/roc_curve.png)


