"""
Client 3: IoT (Sensors)
"""
import torch
import torch.optim as optim
from multi_modal_model import MultiModalFederatedModel
from quantum_e91 import encrypt_data
from data_setup import TabularDataset, get_dynamic_loader
from utils import setup_logger

logger = setup_logger("Client3")

def run_client_3(global_weights, round_num, device, data_path=None):
    logger.info(f"� [Client 3] Training on Sensor Data...")
    
    # Load Data (Tabular)
    dataset = TabularDataset(path=data_path if data_path else './data/table.csv')
    loader = get_dynamic_loader(dataset, round_num)
    
    # Model (Tabular MLP)
    model = MultiModalFederatedModel('tabular').to(device)
    try: model.load_state_dict(global_weights, strict=True)
    except: pass
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(1):
        for dat, lbl in loader:
            dat, lbl = dat.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(dat), lbl)
            loss.backward()
            optimizer.step()
            
    return encrypt_data(model.state_dict())
