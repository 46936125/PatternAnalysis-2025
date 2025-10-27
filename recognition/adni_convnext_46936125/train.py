import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json

from dataset import get_dataloaders, mixup_data, extract_subject_id
from modules import ADNIConvNext

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"convnext_adni_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
config_path = os.path.join(save_dir, "train_config.json")  # Save checkpoint path and threshold

num_classes = 2
batch_size = 32
num_workers = 0
mixup_alpha = 0.5

# Load data
train_loader, val_loader, test_loader = get_dataloaders(data_root="/home/groups/comp3710/ADNI/AD_NC",
                                                        batch_size=batch_size,
                                                        num_workers=num_workers)
print(f"Loaded {len(train_loader.dataset)} training images")
print(f"Loaded {len(val_loader.dataset)} validation images")
print(f"Loaded {len(test_loader.dataset)} test images\n")

# Model setup
model = ADNIConvNext(num_classes=num_classes, dropout_rate=0.5).to(device)
w_nc = 1.0
w_ad = 2.0
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=torch.tensor([w_nc, w_ad]).to(device))

# Training helper
def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    subject_to_probs = {}
    subject_to_labels = {}
    subject_to_count = {}
    loop = tqdm(dataloader, leave=False)

    for batch_idx, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            outputs = model(imgs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Collect per-subject probabilities for accuracy
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        start_idx = batch_idx * dataloader.batch_size
        end_idx = min(start_idx + dataloader.batch_size, len(dataloader.dataset))
        batch_indices = dataloader.dataset.indices[start_idx:end_idx] if hasattr(dataloader.dataset, 'indices') else range(start_idx, end_idx)
        batch_subjects = [extract_subject_id(dataloader.dataset.image_paths[idx])
                         for idx in batch_indices]
        for i, subj_id in enumerate(batch_subjects):
            if subj_id not in subject_to_probs:
                subject_to_probs[subj_id] = []
                subject_to_labels[subj_id] = labels[i].cpu().item()
                subject_to_count[subj_id] = 0
            subject_to_probs[subj_id].append(probs[i])
            subject_to_count[subj_id] += 1

        loop.set_description(f"Train Loss: {loss.item():.4f}")

    # Compute per-subject accuracy
    all_probs, all_labels, all_preds = [], [], []
    for subj_id in subject_to_probs:
        avg_prob = np.mean(subject_to_probs[subj_id])
        pred = 1 if avg_prob > 0.5 else 0
        all_probs.append(avg_prob)
        all_labels.append(subject_to_labels[subj_id])
        all_preds.append(pred)

    subject_acc = accuracy_score(all_labels, all_preds)

    return total_loss / total, subject_acc

def find_optimal_threshold(model, dataloader, device):
    model.eval()
    subject_to_probs = {}
    subject_to_labels = {}
    subject_to_count = {}

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(dataloader.dataset))
            batch_indices = dataloader.dataset.indices[start_idx:end_idx] if hasattr(dataloader.dataset, 'indices') else range(start_idx, end_idx)
            batch_subjects = [extract_subject_id(dataloader.dataset.image_paths[idx])
                             for idx in batch_indices]
            for i, subj_id in enumerate(batch_subjects):
                if subj_id not in subject_to_probs:
                    subject_to_probs[subj_id] = []
                    subject_to_labels[subj_id] = labels[i].cpu().item()
                    subject_to_count[subj_id] = 0
                subject_to_probs[subj_id].append(probs[i])
                subject_to_count[subj_id] += 1

    all_probs, all_labels = [], []
    for subj_id in subject_to_probs:
        avg_prob = np.mean(subject_to_probs[subj_id])
        all_probs.append(avg_prob)
        all_labels.append(subject_to_labels[subj_id])

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_acc, best_threshold = 0, 0.5
    for thresh in thresholds:
        preds = (np.array(all_probs) > thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_threshold

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    subject_to_probs = {}
    subject_to_labels = {}
    subject_to_count = {}

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(dataloader.dataset))
            batch_indices = dataloader.dataset.indices[start_idx:end_idx] if hasattr(dataloader.dataset, 'indices') else range(start_idx, end_idx)
            batch_subjects = [extract_subject_id(dataloader.dataset.image_paths[idx])
                             for idx in batch_indices]
            for i, subj_id in enumerate(batch_subjects):
                if subj_id not in subject_to_probs:
                    subject_to_probs[subj_id] = []
                    subject_to_labels[subj_id] = labels[i].cpu().item()
                    subject_to_count[subj_id] = 0
                subject_to_probs[subj_id].append(probs[i])
                subject_to_count[subj_id] += 1

    all_labels, all_preds, all_probs = [], [], []
    for subj_id in subject_to_probs:
        avg_prob = np.mean(subject_to_probs[subj_id])
        pred = 1 if avg_prob > threshold else 0
        all_labels.append(subject_to_labels[subj_id])
        all_preds.append(pred)
        all_probs.append(avg_prob)

    avg_loss = total_loss / sum(subject_to_count.values())
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, recall, specificity, auc

def plot_metrics(train_losses, val_losses, val_accs, label=""):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label=f"Train Loss {label}")
    plt.plot(val_losses, label=f"Val Loss {label}")
    plt.plot(val_accs, label=f"Val Accuracy {label}")
    plt.legend()
    plt.title(f"Training Progress {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"training_curve_{label}.png"))
    plt.close()

# Instantiate training params
best_val_acc = 0
train_losses, val_losses, val_accs = [], [], []
patience, no_improve_epochs = 20, 0
num_epochs = 150

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=3e-4, step_size_up=5, mode='triangular2')

# Find initial optimal threshold
optimal_threshold = find_optimal_threshold(model, val_loader, device)
print(f"Initial optimal threshold: {optimal_threshold:.4f}")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
    val_loss, val_acc, val_recall, val_spec, val_auc = evaluate(model, val_loader, criterion, device, threshold=optimal_threshold)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Acc (Subject): {train_acc:.4f} | Val Acc (Subject): {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), save_path)
        optimal_threshold = find_optimal_threshold(model, val_loader, device)
        print(f"Best model updated (Val Acc: {val_acc:.4f})")
        print(f"Updated optimal threshold: {optimal_threshold:.4f}")
        with open(config_path, 'w') as f:
            json.dump({"checkpoint_path": save_path, "optimal_threshold": optimal_threshold}, f)
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
        break

plot_metrics(train_losses, val_losses, val_accs)
