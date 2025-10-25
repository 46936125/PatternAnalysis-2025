import os
import torch
from torchvision import transforms
from PIL import Image
from modules import ADNIConvNext
from dataset import ADNIDataset, get_dataloaders
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to trained model checkpoint
checkpoint_path = "checkpoints/convnext_adni_20251008_085822.pth"

# Number of classes
num_classes = 2

# Image transforms (should match validation/test transforms from train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = ADNIConvNext(num_classes=num_classes, pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Prediction functions
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
    """Predict all images in a folder."""
    images = []
    labels = []
    classes = {"NC": 0, "AD": 1}

    for cls_name, cls_idx in classes.items():
        class_dir = os.path.join(folder_path, cls_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(class_dir, fname))
                labels.append(cls_idx)

    all_preds = []
    all_probs = []

    for img_path in images:
        pred, conf = predict_image(img_path)
        all_preds.append(pred)
        all_probs.append(conf)

    return images, labels, all_preds, all_probs

def evaluate_folder(folder_path):
    images, labels, preds, probs = predict_folder(folder_path)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(labels, probs)
    return acc, recall, specificity, auc


if __name__ == "__main__":
    test_folder = "dataset/ADNI/AD_NC/test"

    acc, recall, specificity, auc = evaluate_folder(test_folder)
    print("✅ Test Folder Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC: {auc:.4f}")

    # Predict a single image
    sample_image = "dataset/ADNI/AD_NC/test/AD/388206_78.jpeg"
    pred_class, confidence = predict_image(sample_image)
    label_map = {0: "NC", 1: "AD"}
    print(f"\nSingle Image Prediction: {sample_image}")
    print(f"Predicted Class: {label_map[pred_class]} with confidence {confidence:.4f}")
