import torch
import torch.nn as nn
import torch.optim as optim
from model import UniversalModel
from data_setup import get_hetero_dataloaders
from quantum_e91 import encrypt_data

def run_client_2(global_weights, local_epochs=2, device='cpu'):
    print("\n   [Client 2 - Wildlife Dept] 🐅 Starting Operation...")
    
    (_, train_l2, _), _ = get_hetero_dataloaders()
    
    model = UniversalModel().to(device)  # Move to GPU/CPU
    model.load_state_dict(global_weights)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("   [Client 2] Training on local data...")
    for epoch in range(local_epochs):
        for images, labels in train_l2:
            images = images.to(device)  # Move to GPU/CPU
            labels = labels.to(device)  # Move to GPU/CPU
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
    print("   [Client 2] Training Complete.")
    
    final_weights = model.state_dict()
    encrypted_weights, key = encrypt_data(final_weights)
    print(f"   [Client 2] 🔐 Encrypted with E91 Key: {key[:8]}...")
    
    return encrypted_weights, key