import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from dataset import get_dataloaders, mixup_data
from modules import ADNIConvNext, ModelEMA

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"convnext_adni_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

num_classes = 2
batch_size = 32
num_workers = 0
mixup_alpha = 0.2

# Load data
train_loader, val_loader, test_loader = get_dataloaders(
    data_root="/home/groups/comp3710/ADNI/AD_NC",
    batch_size=batch_size,
    num_workers=num_workers
)

print(f"Loaded {len(train_loader.dataset)} training images")
print(f"Loaded {len(val_loader.dataset)} validation images")
print(f"Loaded {len(test_loader.dataset)} test images\n")

# Model setup
model = ADNIConvNext(num_classes=num_classes, dropout_rate=0.5).to(device)
w_nc = 1.0
w_ad = 1.5
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05,
    weight=torch.tensor([w_nc, w_ad]).to(device)
)

# Training helper
def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, ema=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(dataloader, leave=False)

    for imgs, labels, _ in loop:  # unpack three values
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
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            correct += (preds == labels.cpu().numpy()).sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs)
    recall = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)

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


# Instantiate training parameters
best_val_acc = 0
train_losses, val_losses, val_accs = [], [], []
patience, no_improve_epochs = 15, 0
num_epochs = 200

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
ema = ModelEMA(model, decay=0.999)

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, ema=ema)
    ema.apply_shadow(model)
    val_loss, val_acc, val_recall, val_spec, val_auc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model updated (Val Acc: {val_acc:.4f})")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
        break

plot_metrics(train_losses, val_losses, val_accs)
