"""
Client 2: Security (Text)
"""
import torch
import torch.optim as optim
from multi_modal_model import MultiModalFederatedModel
from quantum_e91 import encrypt_data
from data_setup import TextDataset, get_dynamic_loader
from utils import setup_logger

logger = setup_logger("Client2")

def run_client_2(global_weights, round_num, device, data_path=None):
    logger.info(f"📝 [Client 2] Training on Text Logs...")
    
    # Load Data (Text)
    dataset = TextDataset(path=data_path if data_path else './data/test.txt')
    loader = get_dynamic_loader(dataset, round_num)
    
    # Model (Text LSTM)
    model = MultiModalFederatedModel('text').to(device)
    try: model.load_state_dict(global_weights, strict=True)
    except: pass
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(1):
        for txt, lbl in loader:
            txt, lbl = txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(txt), lbl)
            loss.backward()
            optimizer.step()
            
    return encrypt_data(model.state_dict())
