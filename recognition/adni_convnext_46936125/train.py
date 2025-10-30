import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json

from dataset import get_dataloaders, mixup_data
from modules import ConvNeXtTiny

# ===============================
# Config
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"convnext_adni_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
config_path = os.path.join(save_dir, "train_config.json")  # Save checkpoint path

# ===============================
# Training params
# ===============================
num_classes = 2
batch_size = 128
num_workers = 0
mixup_alpha = 0.2
dropout_rate = 0.5
label_smoothing = 0.1
best_val_acc = 0
train_losses, val_losses, val_accs = [], [], []
patience, no_improve_epochs = 20, 0
num_epochs = 200

# ===============================
# Evaluation params
# ===============================
threshold = 0.5

# ===============================
# Load data
# ===============================
train_loader, val_loader, test_loader = get_dataloaders(data_root="/home/groups/comp3710/ADNI/AD_NC",
                                                        batch_size=batch_size,
                                                        num_workers=num_workers)
print(f"Loaded {len(train_loader.dataset)} training images")
print(f"Loaded {len(val_loader.dataset)} validation images")
print(f"Loaded {len(test_loader.dataset)} test images\n")

# ===============================
# Model setup
# ===============================
model = ConvNeXtTiny(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
w_nc = 1.0
w_ad = 2.0
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=torch.tensor([w_nc, w_ad]).to(device))

# ===============================
# Training helper
# ===============================
def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """
    Train the model for one epoch.
    - Computes loss using MixUp if enabled.
    - Computes subject-level accuracy using clean (non-mixed) images.
    - Works correctly with shuffled dataloaders since subject_ids are returned.
    """
    model.train()
    total_loss, total_samples = 0.0, 0
    subject_to_probs = {}
    subject_to_labels = {}
    subject_to_count = {}
    loop = tqdm(dataloader, leave=False)

    for imgs, labels, subject_ids in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Compute forward for MixUp training loss
        if use_mixup:
            mixed_imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            outputs = model(mixed_imgs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        # Clean forward (no MixUp) for metrics
        with torch.no_grad():
            clean_outputs = model(imgs)
            probs = torch.softmax(clean_outputs, dim=1)[:, 1].detach().cpu().numpy()
            labels_cpu = labels.cpu().numpy()

        # Aggregate per-subject probabilities
        for i, subj_id in enumerate(subject_ids):
            subj_id = str(subj_id)
            if subj_id not in subject_to_probs:
                subject_to_probs[subj_id] = []
                subject_to_labels[subj_id] = labels_cpu[i]
                subject_to_count[subj_id] = 0
            subject_to_probs[subj_id].append(probs[i])
            subject_to_count[subj_id] += 1

        loop.set_description(f"Train Loss: {loss.item():.4f}")

    # Compute subject-level accuracy
    all_probs, all_labels, all_preds = [], [], []
    for subj_id in subject_to_probs:
        avg_prob = np.mean(subject_to_probs[subj_id])
        pred = 1 if avg_prob > threshold else 0
        all_probs.append(avg_prob)
        all_labels.append(subject_to_labels[subj_id])
        all_preds.append(pred)

    subject_acc = accuracy_score(all_labels, all_preds)

    return total_loss / total_samples, subject_acc

# ===============================
# Evaluation helper
# ===============================
def evaluate(model, dataloader, criterion, device, threshold=threshold):
    """
    Evaluate model at the subject level.
    - Collects per-subject probabilities.
    - Computes subject-level metrics: loss, accuracy, recall, specificity, AUC.
    - Works correctly even if dataloader is shuffled (uses subject_ids).
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    subject_to_probs = {}
    subject_to_labels = {}
    subject_to_count = {}

    with torch.no_grad():
        for imgs, labels, subject_ids in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            for i, subj_id in enumerate(subject_ids):
                subj_id = str(subj_id)
                if subj_id not in subject_to_probs:
                    subject_to_probs[subj_id] = []
                    subject_to_labels[subj_id] = labels_cpu[i]
                    subject_to_count[subj_id] = 0
                subject_to_probs[subj_id].append(probs[i])
                subject_to_count[subj_id] += 1

    # Compute per-subject metrics
    all_labels, all_preds, all_probs = [], [], []
    for subj_id in subject_to_probs:
        avg_prob = np.mean(subject_to_probs[subj_id])
        pred = 1 if avg_prob > threshold else 0
        all_labels.append(subject_to_labels[subj_id])
        all_preds.append(pred)
        all_probs.append(avg_prob)

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, recall, specificity, auc, all_labels, all_preds, all_probs

# ===============================
# Plotting helpers to showcase model output
# ===============================
def plot_metrics(train_losses, val_losses, val_accs):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label=f"Train Loss")
    plt.plot(val_losses, label=f"Val Loss")
    plt.plot(val_accs, label=f"Val Accuracy")
    plt.legend()
    plt.title(f"Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"training_curve.png"))
    plt.close()

def plot_confusion_matrix(all_labels, all_preds):
    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Subject-Level)")
    plt.grid(False)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(all_labels, all_probs):
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title("ROC Curve (Subject-Level)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()


# ===============================
# Training
# ===============================
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
    val_loss, val_acc, val_recall, val_spec, val_auc, all_labels, all_preds, all_probs = evaluate(model, 
                                                                                                  val_loader, 
                                                                                                  criterion, 
                                                                                                  device, 
                                                                                                  threshold=threshold)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Acc (Subject): {train_acc:.4f} | Val Acc (Subject): {val_acc:.4f}")

    # Use highest validation accuracy as the chosen model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model updated (Val Acc: {val_acc:.4f})")

        with open(config_path, 'w') as f:
            json.dump({"checkpoint_path": save_path}, f)
    else:
        no_improve_epochs += 1

plot_metrics(train_losses, val_losses, val_accs)
plot_confusion_matrix(all_labels, all_preds)
plot_roc_curve(all_labels, all_probs)
