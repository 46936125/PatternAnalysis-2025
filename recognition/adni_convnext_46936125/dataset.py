import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np


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

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Label mapping
        self.classes = {'NC': 0, 'AD': 1}

        self.image_paths = []
        self.labels = []

        for label_name, label_idx in self.classes.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label


def get_dataloaders(data_root, batch_size=32, num_workers=0, val_split=0.15, seed=42):
    """
    Creates train, validation, and test dataloaders from ADNIDataset.

    Folder structure:
        dataset/ADNI/AD_NC/
            train/
                AD/
                NC/
            test/
                AD/
                NC/
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Training augmentations for generalisation
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.1156, 0.1156, 0.1156], [0.2198, 0.2198, 0.2198])
    ])

    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1156, 0.1156, 0.1156], [0.2198, 0.2198, 0.2198])
    ])

    # Load datasets
    full_train_dataset = ADNIDataset(train_dir, transform=train_transform)
    test_dataset = ADNIDataset(test_dir, transform=val_test_transform)

    # Manually create a validation split from train
    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    val_dataset.dataset.transform = val_test_transform


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataloaders ready:")
    print(f"Train: {train_size} images")
    print(f"Val:   {val_size} images (split from train)")
    print(f"Test:  {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


# mixup
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