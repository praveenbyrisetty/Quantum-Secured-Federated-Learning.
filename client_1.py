"""
Client 1: Traffic (Images)
"""
import torch
import torch.optim as optim
from multi_modal_model import MultiModalFederatedModel
from quantum_e91 import encrypt_data
from data_setup import get_image_dataset, get_dynamic_loader
from utils import setup_logger

logger = setup_logger("Client1")

def run_client_1(global_weights, round_num, device, data_path=None):
    logger.info(f"� [Client 1] Training on Images...")
    
    # Load Data (Images)
    dataset = get_image_dataset(path=data_path if data_path else './data/images')
    loader = get_dynamic_loader(dataset, round_num)
    
    # Model (Image CNN)
    model = MultiModalFederatedModel('image').to(device)
    try: model.load_state_dict(global_weights, strict=True)
    except: pass # Skip if mismatched (e.g. first round)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(1): # 1 Epoch
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), lbl)
            loss.backward()
            optimizer.step()
            
    return encrypt_data(model.state_dict())
