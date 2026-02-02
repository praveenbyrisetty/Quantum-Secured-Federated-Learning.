import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. IMAGE DATASET (Traffic)
# ==========================================
def get_image_dataset(path='./data/images'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if os.path.exists(path) and len(os.listdir(path)) > 0:
        try:
            full_data = torchvision.datasets.ImageFolder(root=path, transform=transform)
            return full_data
        except:
            pass
            
    # Default Fallback: CIFAR-10 subset (Vehicles)
    full_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    idx = [i for i, t in enumerate(full_data.targets) if t in [0, 1, 8, 9]] # Plane, Car, Ship, Truck
    
    class Remapped(Dataset):
        def __init__(self, s): self.s=s; self.m={0:0, 1:1, 8:2, 9:3}
        def __len__(self): return len(self.s)
        def __getitem__(self, i): img,l = self.s[i]; return img, self.m[l]
        
    return Remapped(Subset(full_data, idx))

# ==========================================
# 2. TEXT DATASET (Security Logs)
# ==========================================
class TextDataset(Dataset):
    def __init__(self, size=1000, path='./data/test.txt'):
         self.data = []
         self.labels = []
         
         # Try loading real
         if os.path.exists(path):
             try:
                 with open(path, 'r') as f: lines = f.readlines()
                 for _ in range(size):
                     line = lines[np.random.randint(0, len(lines))]
                     # Hash words to IDs
                     words = line.split()
                     ids = [hash(w) % 5000 for w in words[:20]]
                     if len(ids) < 20: ids += [0]*(20-len(ids))
                     self.data.append(torch.tensor(ids).long())
                     self.labels.append(1 if "error" in line else 0)
             except: pass
             
         # Fallback Synthetic
         if not self.data:
             for _ in range(size):
                 self.data.append(torch.randint(0, 5000, (20,)))
                 self.labels.append(torch.randint(0, 2, (1,)).item())
        
         self.labels = torch.tensor(self.labels).long()

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

# ==========================================
# 3. TABULAR DATASET (IoT Sensors)
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, size=1000, path='./data/table.csv'):
        # Try loading real
        loaded = False
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                real_data = torch.tensor(df.iloc[:, :-1].values).float()
                real_labels = torch.tensor(df.iloc[:, -1].values).long()
                # Supersample
                idx = np.random.randint(0, len(real_data), size)
                self.data = real_data[idx]
                self.labels = real_labels[idx]
                loaded = True
            except: pass
            
        if not loaded:
            self.data = torch.randn(size, 10)
            self.labels = torch.randint(0, 3, (size,))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

#Helpers
def get_dynamic_loader(dataset, round_num, chunk_size=1000, batch_size=32):
    total = len(dataset)
    start = (round_num-1)*chunk_size % total
    end = min(start+chunk_size, total)
    return DataLoader(Subset(dataset, range(start, end)), batch_size=batch_size, shuffle=True)