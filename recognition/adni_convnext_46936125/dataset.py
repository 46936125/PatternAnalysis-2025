import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

class ADNIDataset(Dataset):
    """
    PyTorch Dataset for loading ADNI MRI slice images.
    Expected structure:
        root/
            AD/
            NC/
    - AD : Alzheimer’s disease (label = 1)
    - NC : Normal Control (label = 0)
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.indices = list(range(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # return subject id so callers do not have to reconstruct indices
        subject_id = extract_subject_id(img_path)
        return image, label, subject_id
    

def extract_subject_id(filename: str) -> str:
    """
    Extracts the subject ID from a filename like '218391_78.jpeg' -> '218391'.
    """
    base = os.path.basename(filename)
    return base.split('_')[0]

def get_dataloaders(data_root, batch_size=32, num_workers=0, val_split=0.15, seed=42):
    """
    Creates train, validation, and test dataloaders for ADNI MRI data.
    Splits the training data by *subject ID* to prevent data leakage.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.1156, 0.1156, 0.1156],
                             [0.2198, 0.2198, 0.2198])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1156, 0.1156, 0.1156],
                             [0.2198, 0.2198, 0.2198])
    ])

    # Collate all images
    classes = {'NC': 0, 'AD': 1}
    all_paths, all_labels, all_subjects = [], [], []

    for label_name, label_idx in classes.items():
        class_dir = os.path.join(train_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(class_dir, fname)
                subject_id = extract_subject_id(fname)
                all_paths.append(path)
                all_labels.append(label_idx)
                all_subjects.append(subject_id)

    print(f"Loaded {len(all_paths)} training images from {train_dir}")

    # Split subjects with K fold
    sgkf = StratifiedGroupKFold(n_splits=int(1 / val_split),
                                shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(all_paths, all_labels, groups=all_subjects))

    def subset(indices):
        return [all_paths[i] for i in indices], [all_labels[i] for i in indices]

    train_paths, train_labels = subset(train_idx)
    val_paths, val_labels = subset(val_idx)

    train_subjects = set([extract_subject_id(p) for p in train_paths])
    val_subjects = set([extract_subject_id(p) for p in val_paths])
    overlap = train_subjects.intersection(val_subjects)
    assert len(overlap) == 0, f"Subject leakage detected: {overlap}"

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Train images:   {len(train_paths)}")
    print(f"Val images:     {len(val_paths)}")

    train_dataset = ADNIDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ADNIDataset(val_paths, val_labels, transform=val_test_transform)

    test_paths, test_labels = [], []
    for label_name, label_idx in classes.items():
        class_dir = os.path.join(test_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_paths.append(os.path.join(class_dir, fname))
                test_labels.append(label_idx)
    test_dataset = ADNIDataset(test_paths, test_labels, transform=val_test_transform)

    print(f"Test images:    {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataloaders ready:")
    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    return train_loader, val_loader, test_loader

# Mixup
def mixup_data(x, y, alpha=0.2):
    """Applies Mixup data augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam