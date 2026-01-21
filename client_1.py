import torch
import torch.nn as nn
import torch.optim as optim
from model import UniversalModel
from data_setup import get_hetero_dataloaders
from quantum_e91 import encrypt_data

def run_client_1(global_weights, local_epochs=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n   [Client 1 - Traffic Dept] 🚗 Starting on {device}...")
    
    (train_l1, _, _), _ = get_hetero_dataloaders()
    model = UniversalModel().to(device)                 # ← GPU
    model.load_state_dict(global_weights)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("   [Client 1] Training...")
    for epoch in range(local_epochs):
        for images, labels in train_l1:
            images = images.to(device)                  # ← GPU
            labels = labels.to(device)                  # ← GPU
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    print("   [Client 1] Training Complete.")
    final_weights = model.state_dict()
    encrypted_weights, key = encrypt_data(final_weights)
    print(f"   [Client 1] 🔐 Encrypted with E91 Key: {key[:8]}...")
    
    return encrypted_weights, key