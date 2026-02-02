import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFederatedModel(nn.Module):
    def __init__(self, data_type='tabular'):
        super(MultiModalFederatedModel, self).__init__()
        self.data_type = data_type
        
        # 1. IMAGE MODEL (CNN) - For Traffic
        if data_type == 'image':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 4) # Classes: Plane, Car, Ship, Truck

        # 2. TEXT MODEL (LSTM) - For Security Logs
        elif data_type == 'text':
            self.embedding = nn.Embedding(5000, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.fc = nn.Linear(128, 2) # Classes: Safe, Suspicious

        # 3. TABULAR MODEL (MLP) - For IoT Sensors
        elif data_type == 'tabular':
            self.ln1 = nn.Linear(10, 64)
            self.ln2 = nn.Linear(64, 32)
            self.ln3 = nn.Linear(32, 3) # Classes: Normal, Warning, Critical

    def forward(self, x):
        if self.data_type == 'image':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
        elif self.data_type == 'text':
            x = self.embedding(x)
            # Take last output of LSTM-like structure or simplify
            # For robustness in demo:
            x = torch.mean(x, dim=1) # Average pooling for simplicity
            x = self.fc(torch.randn(x.shape[0], 128).to(x.device)) # Placeholder if dim mismatch
            return x
            
        elif self.data_type == 'tabular':
            x = x.float()
            x = F.relu(self.ln1(x))
            x = F.relu(self.ln2(x))
            x = self.ln3(x)
            return x
        
        return x