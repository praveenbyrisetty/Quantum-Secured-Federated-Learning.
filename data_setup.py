"""
FLQC Data Setup - HAM10000 Skin Lesion Dataset

WHAT THIS FILE DOES:
Loads the HAM10000 dermoscopic skin lesion dataset and partitions it
across 3 federated learning clients (simulating different hospitals).

DATASET: HAM10000 — 10,015 images, 7 classes:
  - nv:    Melanocytic Nevi (benign moles)
  - mel:   Melanoma (malignant)
  - bkl:   Benign Keratosis
  - bcc:   Basal Cell Carcinoma (malignant)
  - akiec: Actinic Keratoses (pre-malignant)
  - vasc:  Vascular Lesions
  - df:    Dermatofibroma

CLIENT PARTITIONING (non-IID, simulating hospital specialization):
  - Client 0 (Hospital A): nv, mel   — Melanocytic focus
  - Client 1 (Hospital B): bkl, bcc, akiec — Keratosis & Carcinoma
  - Client 2 (Hospital C): vasc, df  — Vascular & Fibroma

SETUP:
  Download HAM10000 from Kaggle and place in ./data/HAM10000/:
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

  Expected structure after setup:
  ./data/HAM10000/
      HAM10000_metadata.csv
      HAM10000_images_part_1/
      HAM10000_images_part_2/
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import glob
import shutil
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
HAM10000_DIR = os.path.join('.', 'data', 'HAM10000')
ORGANIZED_DIR = os.path.join('.', 'data', 'HAM10000_organized')
IMAGE_SIZE = 128  # Resize dermoscopy images to 128x128 (balance speed vs quality)

# 7 HAM10000 classes
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Friendly display names
CLASS_DISPLAY = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions',
}

# Client class assignments (non-IID partitioning)
CLIENT_CLASSES = {
    0: ['nv', 'mel'],           # Hospital A: Melanocytic focus
    1: ['bkl', 'bcc', 'akiec'], # Hospital B: Keratosis & Carcinoma
    2: ['vasc', 'df'],          # Hospital C: Vascular & Fibroma
}


# ==========================================
# 1. ORGANIZE DATASET INTO CLASS FOLDERS
# ==========================================
def organize_ham10000():
    """
    Read metadata CSV and copy images into class-specific folders.
    Creates: ./data/HAM10000_organized/<class_name>/<image>.jpg
    
    This only needs to run once. Subsequent calls skip if already done.
    """
    if os.path.exists(ORGANIZED_DIR) and len(os.listdir(ORGANIZED_DIR)) >= 7:
        return True  # Already organized
    
    # Find metadata CSV
    csv_path = None
    for candidate in [
        os.path.join(HAM10000_DIR, 'HAM10000_metadata.csv'),
        os.path.join(HAM10000_DIR, 'HAM10000_metadata'),
        os.path.join(HAM10000_DIR, 'hmnist_28_28_RGB.csv'),
    ]:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    
    # Also search recursively
    if csv_path is None:
        for root, dirs, files in os.walk(HAM10000_DIR):
            for f in files:
                if 'metadata' in f.lower() and f.endswith('.csv'):
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break
    
    if csv_path is None:
        print(f"ERROR: Cannot find HAM10000_metadata.csv in {HAM10000_DIR}")
        print("Please download the HAM10000 dataset from:")
        print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"And place it in: {os.path.abspath(HAM10000_DIR)}")
        return False
    
    print(f"Found metadata: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find all image files
    image_dirs = []
    for root, dirs, files in os.walk(HAM10000_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_dirs.append(root)
                break
    
    if not image_dirs:
        print(f"ERROR: No images found in {HAM10000_DIR}")
        return False
    
    # Build image path lookup
    image_paths = {}
    for img_dir in image_dirs:
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(f)[0]
                image_paths[name] = os.path.join(img_dir, f)
    
    print(f"Found {len(image_paths)} images")
    
    # Create class folders and copy images
    os.makedirs(ORGANIZED_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(ORGANIZED_DIR, class_name), exist_ok=True)
    
    copied = 0
    for _, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']  # diagnosis class
        
        if dx not in CLASS_NAMES:
            continue
        
        if image_id in image_paths:
            src = image_paths[image_id]
            ext = os.path.splitext(src)[1]
            dst = os.path.join(ORGANIZED_DIR, dx, f"{image_id}{ext}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1
    
    print(f"Organized {copied} images into {len(CLASS_NAMES)} class folders")
    return True


# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class HAM10000Dataset(Dataset):
    """
    HAM10000 Skin Lesion Dataset.
    
    Loads images from organized class folders and applies transforms.
    Supports filtering to specific classes for FL client partitioning.
    """
    
    def __init__(self, root_dir, transform=None, selected_classes=None):
        """
        Args:
            root_dir: Path to organized dataset (HAM10000_organized/)
            transform: Image transforms to apply
            selected_classes: List of class names to include (None = all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, label_idx)
        
        classes = selected_classes if selected_classes else CLASS_NAMES
        
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            label_idx = CLASS_TO_IDX[class_name]
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_file),
                        label_idx
                    ))
        
        # Shuffle for good measure
        np.random.seed(42)
        np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==========================================
# 3. DATA LOADING FUNCTIONS
# ==========================================

def get_transforms(train=True):
    """Get image transforms for HAM10000 dermoscopic images."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7635, 0.5461, 0.5705],  # HAM10000-specific means
                std=[0.1409, 0.1520, 0.1695]     # HAM10000-specific stds
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7635, 0.5461, 0.5705],
                std=[0.1409, 0.1520, 0.1695]
            )
        ])


def get_client_dataset(client_id, total_clients=3, train=True):
    """
    Get dataset for each client based on non-IID class partitioning.
    
    - Client 0 (Hospital A): nv, mel — Melanocytic focus
    - Client 1 (Hospital B): bkl, bcc, akiec — Keratosis & Carcinoma
    - Client 2 (Hospital C): vasc, df — Vascular & Fibroma
    
    Args:
        client_id: Integer ID of the client (0, 1, 2)
        total_clients: Total number of clients (default 3)
        train: Whether to use training set (True) or test set (False)
    
    Returns:
        Dataset partition for this client
    """
    # Organize dataset if not already done
    if not organize_ham10000():
        raise FileNotFoundError(
            f"HAM10000 dataset not found in {HAM10000_DIR}. "
            "Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
        )
    
    # Get this client's assigned classes
    selected_classes = CLIENT_CLASSES.get(client_id, CLASS_NAMES)
    
    # Load full dataset for this client's classes
    transform = get_transforms(train=train)
    full_dataset = HAM10000Dataset(
        root_dir=ORGANIZED_DIR,
        transform=transform,
        selected_classes=selected_classes
    )
    
    if len(full_dataset) == 0:
        raise ValueError(
            f"No images found for Client {client_id} "
            f"(classes: {selected_classes}). "
            f"Check that {ORGANIZED_DIR} contains the class folders."
        )
    
    # 80/20 train/test split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    return train_set if train else test_set


def get_full_test_dataset():
    """
    Get the full test set across ALL classes (for global model evaluation).
    Returns a dataset with all 7 classes.
    """
    if not organize_ham10000():
        raise FileNotFoundError("HAM10000 dataset not found.")
    
    transform = get_transforms(train=False)
    full_dataset = HAM10000Dataset(
        root_dir=ORGANIZED_DIR,
        transform=transform,
        selected_classes=None  # All classes
    )
    
    # Use 20% as test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    _, test_set = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    return test_set


# Helper for dynamic loading
def get_dynamic_loader(dataset, round_num, chunk_size=1000, batch_size=32):
    total = len(dataset)
    start = (round_num - 1) * chunk_size % total
    end = min(start + chunk_size, total)
    return DataLoader(Subset(dataset, range(start, end)), batch_size=batch_size, shuffle=True)