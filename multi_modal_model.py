"""
FLQC Image Classification Model
CNN for heterogeneous federated learning (CIFAR-10 + MNIST)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFederatedModel(nn.Module):
    """CNN Model for CIFAR-10 Image Classification (10 classes)"""
    def __init__(self, data_type='image'):
        super(MultiModalFederatedModel, self).__init__()
        
        # CNN Architecture for 32x32 RGB images
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv Block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv Block 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 8x8 -> 4x4
        x = self.dropout(x)
        
        # Flatten and FC layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x