"""
FLQC Image Classification Model
CNN for Federated Learning on HAM10000 Skin Lesion Dataset

Architecture: 3 Conv blocks (Conv + BN + ReLU + Pool) → 3 FC layers
Input: 128x128 RGB dermoscopic images (3 channels)
Output: 7 class logits (HAM10000 skin lesion types)

Classes:
  0: akiec (Actinic Keratoses)
  1: bcc   (Basal Cell Carcinoma)
  2: bkl   (Benign Keratosis)
  3: df    (Dermatofibroma)
  4: mel   (Melanoma)
  5: nv    (Melanocytic Nevi)
  6: vasc  (Vascular Lesions)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 7  # HAM10000 has 7 skin lesion classes

class QuantumE91Model(nn.Module):
    """
    CNN Model for HAM10000 Skin Lesion Classification (7 classes).
    
    Architecture: 3 Conv blocks (Conv + BN + ReLU + Pool) → 3 FC layers
    Input: 128x128 RGB images (3 channels)
    Output: 7 class logits
    
    After 3 MaxPool layers: 128 → 64 → 32 → 16
    So FC input = 128 channels × 16 × 16 = 32,768
    """
    def __init__(self, data_type='image'):
        super(QuantumE91Model, self).__init__()
        
        # Conv Block 1: 3 channels → 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: 32 channels → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3: 64 channels → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Shared pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # FC layers: 128 * 16 * 16 = 32768 → 512 → 128 → 7
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # Conv Block 1: Conv → BN → ReLU → Pool
        # 128x128 → 64x64
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2: Conv → BN → ReLU → Pool
        # 64x64 → 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3: Conv → BN → ReLU → Pool → Dropout
        # 32x32 → 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten and FC layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
