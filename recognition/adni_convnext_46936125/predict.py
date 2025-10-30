import os
import torch
from torchvision import transforms
from PIL import Image
from modules import ConvNeXtTiny
from dataset import extract_subject_id
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
import json

# ===============================
# Config
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load checkpoint path and threshold from train.py
config_path = "checkpoints/train_config.json"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file {config_path} not found. Run train.py first.")

with open(config_path, 'r') as f:
    config = json.load(f)
    checkpoint_path = config["checkpoint_path"]
    print(f"Loaded checkpoint: {checkpoint_path}")

num_classes = 2 # AD / NC

# Image transforms (matching val/test transforms from dataset.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.1156, 0.1156, 0.1156],
                         [0.2198, 0.2198, 0.2198])
])

# ===============================
# Load model from modules.py
# ===============================
model = ConvNeXtTiny(num_classes=num_classes, dropout_rate=0.5)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# ===============================
# Prediction functions for single image / entire folder
# ===============================
def predict_image(image_path):
    """Predict AD vs NC for a single image."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence

def predict_folder(folder_path):
    """Predict all images in a folder, grouping by subject."""
    images = []
    labels = []
    subject_ids = []
    classes = {"NC": 0, "AD": 1}

    for cls_name, cls_idx in classes.items():
        class_dir = os.path.join(folder_path, cls_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(class_dir, fname))
                labels.append(cls_idx)
                subject_ids.append(extract_subject_id(fname))

    subject_to_slices = {}
    subject_to_label = {}
    for img_path, label, subject_id in zip(images, labels, subject_ids):
        if subject_id not in subject_to_slices:
            subject_to_slices[subject_id] = []
            subject_to_label[subject_id] = label
        subject_to_slices[subject_id].append(img_path)

    all_predictions = []
    all_probabilities = []
    all_labels = []

    for subject_id, img_paths in subject_to_slices.items():
        subject_probabilities = []
        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                probabilities = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                subject_probabilities.append(probabilities[0])
        
        avg_probability = np.mean(subject_probabilities)
        pred_class = 1 if avg_probability > 0.5 else 0
        all_predictions.append(pred_class)
        all_probabilities.append(avg_probability)
        all_labels.append(subject_to_label[subject_id])

    return list(subject_to_slices.keys()), all_labels, all_predictions, all_probabilities

def evaluate_folder(folder_path):
    """Evaluate subject-level metrics for a folder."""
    subjects, labels, preds, probs = predict_folder(folder_path)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(labels, probs)
    return acc, recall, specificity, auc

# ===============================
# Display performance of the model on test set
# ===============================
if __name__ == "__main__":
    test_folder = "/home/groups/comp3710/ADNI/AD_NC/test"

    acc, recall, specificity, auc = evaluate_folder(test_folder)
    print("Test Folder Evaluation (Subject-Level):")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Correct AD classification accuracy: {recall:.4f}")
    print(f"Correct NC classification accuracy: {specificity:.4f}")
    print(f"AUC: {auc:.4f}")

    # Predict a single image
    sample_image = "/home/groups/comp3710/ADNI/AD_NC/test/AD/388206_78.jpeg"
    pred_class, confidence = predict_image(sample_image)
    label_map = {0: "NC", 1: "AD"}
    print(f"\nSingle Image Prediction: {sample_image}")
    print(f"Predicted Class: {label_map[pred_class]} with confidence {confidence:.4f}")