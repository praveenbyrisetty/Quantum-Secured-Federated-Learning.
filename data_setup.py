import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# ==========================================
# 1. CLASS-BASED CIFAR-10 PARTITIONING FOR HETEROGENEOUS FL
# ==========================================
def get_partitioned_cifar10(client_id, total_clients=3, train=True):
    """
    Get class-based partitions of CIFAR-10 for heterogeneous FL.
    
    CIFAR-10 Classes:
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    
    Distribution:
    - Client 0 (displayed as Client 1): Non-living things (airplane, automobile, ship, truck)
    - Client 1 (displayed as Client 2): Living things (bird, cat, deer, dog, frog, horse)
    - Client 2 (displayed as Client 3): MNIST (handled by get_client_dataset)
    
    Args:
        client_id: Integer ID of the client (0, 1, 2)
        total_clients: Total number of clients (default 3)
        train: Whether to use training set (True) or test set (False)
    
    Returns:
        Dataset partition for this client
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download full CIFAR-10
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    # Define class assignments for each client
    if client_id == 0:
        # Client 0 (Client 1 in UI): Non-living things
        selected_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
    elif client_id == 1:
        # Client 1 (Client 2 in UI): All living things
        selected_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
    else:
        # Client 2 and beyond: will use MNIST (not CIFAR-10)
        # This is a fallback that shouldn't be used
        selected_classes = []
    
    # Filter dataset to only include selected classes
    indices = [i for i, (_, label) in enumerate(full_dataset) if label in selected_classes]
    partition = Subset(full_dataset, indices)
    
    return partition


def get_mnist_as_cifar_format(train=True):
    """
    Load MNIST and preprocess to match CIFAR-10 format (32x32 RGB).
    Converts grayscale 28x28 digits to 32x32 RGB to work with CIFAR-10 model.
    
    Args:
        train: Whether to use training set (True) or test set (False)
    
    Returns:
        MNIST dataset with CIFAR-10 compatible preprocessing
    """
    transform = transforms.Compose([
        transforms.Pad(2),  # 28x28 → 32x32 by adding 2px padding on each side
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale → RGB (replicate channel)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )


def get_client_dataset(client_id, total_clients=3, train=True):
    """
    Get dataset for each client based on heterogeneous FL setup.
    
    - Client 0 (displayed as Client 1): CIFAR-10 Non-living (airplane, automobile, ship, truck)
    - Client 1 (displayed as Client 2): CIFAR-10 Living (bird, cat, deer, dog, frog, horse)
    - Client 2 (displayed as Client 3): MNIST Digits (0-9)
    
    Args:
        client_id: Integer ID of the client (0, 1, 2)
        total_clients: Total number of clients
        train: Whether to use training set (True) or test set (False)
    
    Returns:
        Dataset for this client
    """
    if client_id == 2:
        # Client 2 (Client 3 in UI): MNIST digits
        return get_mnist_as_cifar_format(train=train)
    else:
        # Client 0 & 1: CIFAR-10 partitions
        return get_partitioned_cifar10(client_id, total_clients, train=train)


def get_image_dataset(path='./data/images'):
    """Legacy function - now defaults to CIFAR-10"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Explicit trigger for CIFAR-10
    if path == 'CIFAR10':
        pass # Skip to fallback
    elif os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
        try:
            full_data = torchvision.datasets.ImageFolder(root=path, transform=transform)
            return full_data
        except:
            pass
            
    # Default Fallback: CIFAR-10 full (10 classes)
    full_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return full_data

# ==========================================
# 2. TEXT DATASET (Security Logs)
# ==========================================
class TextDataset(Dataset):
    def __init__(self, size=1000, path='./data/test.txt'):
         self.data = []
         self.labels = []
         self.raw_data = []
         
         # Try loading real
         if os.path.exists(path):
             try:
                 with open(path, 'r') as f: lines = f.readlines()
                 for _ in range(size):
                     line = lines[np.random.randint(0, len(lines))]
                     self.raw_data.append(line.strip())
                     # Hash words to IDs
                     words = line.split()
                     ids = [hash(w) % 5000 for w in words[:20]]
                     if len(ids) < 20: ids += [0]*(20-len(ids))
                     self.data.append(torch.tensor(ids).long())
                     self.labels.append(1 if "error" in line else 0)
             except: pass
             
         # Fallback Synthetic
         if not self.data:
             for k in range(size):
                 from random import randint
                 self.raw_data.append(f"Synthetic Log Entry {k}: System check {randint(100,999)}")
                 self.data.append(torch.randint(0, 5000, (20,)))
                 self.labels.append(torch.randint(0, 2, (1,)).item())
        
         self.labels = torch.tensor(self.labels).long()

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]
    def get_raw(self, i): return self.raw_data[i]

#Helpers
def get_dynamic_loader(dataset, round_num, chunk_size=1000, batch_size=32):
    total = len(dataset)
    start = (round_num-1)*chunk_size % total
    end = min(start+chunk_size, total)
    return DataLoader(Subset(dataset, range(start, end)), batch_size=batch_size, shuffle=True)