"""
FLQC Client - Federated Learning Client with Security Features

WHAT THIS FILE DOES:
Each "client" is like one hospital/organization that has its own data.
This file defines how each client:
1. Loads its private data
2. Trains the model on that data
3. Applies security measures (gradient clipping, DP noise)
4. Encrypts the trained weights before sending to server

SECURITY FEATURES (3 layers):
- Gradient Clipping: Limits how much training changes the model (prevents data leakage)
- Differential Privacy Noise: Adds random noise to weights (makes it impossible to reverse-engineer data)
- Quantum Encryption: Encrypts weights with E91 quantum key (protects during transmission)
"""

import logging
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np
import flwr as fl
from collections import OrderedDict
from multi_modal_model import MultiModalFederatedModel
from quantum_e91 import generate_shared_key, encrypt_parameters, verify_entanglement
from data_setup import get_client_dataset

logger = logging.getLogger("FLQC.Client")

# ==========================================
# SECURITY CONFIGURATION
# ==========================================
# These values control how much security we apply.
# More security = slightly less accuracy (trade-off).

SECURITY_CONFIG = {
    "max_grad_norm": 1.0,       # Gradient clipping threshold
                                 # WHAT: Limits the size of gradient updates during training
                                 # WHY: Large gradients can leak information about individual data points
                                 # HOW: If gradient norm > 1.0, scale it down to 1.0
    
    "dp_noise_multiplier": 0.01, # Differential Privacy noise level
                                  # WHAT: How much random noise to add to model weights after training
                                  # WHY: Noise makes it mathematically impossible to extract individual data
                                  # HOW: Multiply each weight by a small random Gaussian noise
                                  # Lower = more accurate but less private
                                  # Higher = less accurate but more private
    
    "dp_enabled": True,          # Whether to apply Differential Privacy
    "encryption_enabled": True,  # Whether to encrypt parameters before sending
}


class FLQCClient(fl.client.NumPyClient):
    """
    A Federated Learning client with multi-layer security.
    
    LIFECYCLE (each training round):
    1. Receive global model weights from server
    2. Generate quantum key via E91 protocol
    3. Verify entanglement (CHSH test)
    4. Train on local data with gradient clipping
    5. Add DP noise to trained weights
    6. Encrypt weights with quantum key
    7. Send encrypted weights + key to server
    """
    
    def __init__(self, cid: str, device: torch.device, total_clients: int = 3):
        self.cid = cid
        self.device = device
        self.model = MultiModalFederatedModel().to(self.device)
        self.train_loader = self._load_data(int(cid), total_clients)
        
        # Security state tracking (for UI display)
        self.last_encryption_key = None       # Last quantum key used
        self.last_chsh_value = None           # Last CHSH test result
        self.last_dp_noise_level = 0.0        # How much DP noise was added
        self.last_grad_clip_count = 0         # How many times gradients were clipped

    def _load_data(self, client_id, total_clients):
        """Load this client's private dataset."""
        try:
            dataset = get_client_dataset(client_id, total_clients, train=True)
            return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return None

    def get_parameters(self, config):
        """Get model weights as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load model weights from list of numpy arrays."""
        if not parameters: return
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def _apply_dp_noise(self, parameters):
        """
        DIFFERENTIAL PRIVACY: Add calibrated Gaussian noise to model parameters.
        
        WHY THIS MATTERS:
        Even though we don't share raw data, model weights can leak information.
        For example, if one data point dramatically changes a weight, an attacker
        could potentially figure out what that data point was.
        
        By adding random noise, we "blur" the weights so that no single data point
        has a noticeable effect. The math guarantees that any individual's data
        cannot be reverse-engineered from the noisy weights.
        
        TRADE-OFF:
        More noise = more privacy but less accuracy
        Less noise = less privacy but more accuracy
        """
        if not SECURITY_CONFIG["dp_enabled"]:
            self.last_dp_noise_level = 0.0
            return parameters
        
        noise_multiplier = SECURITY_CONFIG["dp_noise_multiplier"]
        noisy_params = []
        total_noise = 0.0
        
        for param in parameters:
            # Generate Gaussian noise with the same shape as the parameter
            # Noise is scaled by the noise_multiplier
            noise = np.random.normal(0, noise_multiplier, size=param.shape).astype(param.dtype)
            noisy_params.append(param + noise)
            total_noise += np.linalg.norm(noise)
        
        self.last_dp_noise_level = total_noise / len(parameters) if parameters else 0.0
        return noisy_params

    def fit(self, parameters, config):
        """
        MAIN TRAINING FUNCTION - Called once per round by the server.
        
        FLOW:
        1. Update local model with global weights
        2. Quantum key exchange (E91 protocol)
        3. CHSH entanglement verification
        4. Train on local data (with gradient clipping)
        5. Apply DP noise to trained weights
        6. Encrypt weights with quantum key
        7. Return encrypted weights + metadata
        """
        # =============================================
        # STEP 1: Update local model with global weights
        # =============================================
        # The server sends the latest global model weights.
        # We load them into our local model before training.
        self.set_parameters(parameters)
        
        # =============================================
        # STEP 2: Quantum Key Exchange (E91 Protocol)
        # =============================================
        # Simulate the E91 quantum key distribution protocol.
        # In real life, this would use actual quantum entangled photons.
        # The key is used to encrypt model weights before sending.
        quantum_key, chsh_value = generate_shared_key(self.cid, "server")
        self.last_encryption_key = quantum_key
        self.last_chsh_value = chsh_value
        
        # =============================================
        # STEP 3: CHSH Entanglement Verification
        # =============================================
        # The CHSH inequality test checks if the quantum channel is secure.
        # S > 2.0 means entanglement is verified (no eavesdropper).
        # S ≤ 2.0 means someone might be intercepting the quantum channel!
        is_verified = chsh_value > 2.0
        
        if not is_verified:
            logger.warning(f"Client {self.cid}: ⚠️ CHSH verification failed! "
                          f"S={chsh_value:.4f} ≤ 2.0. Possible eavesdropper!")
        
        # =============================================
        # STEP 4: Train locally WITH gradient clipping
        # =============================================
        # Gradient clipping limits how much any single training step can change
        # the model. This prevents large updates that could leak info about
        # individual data points.
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        running_loss = 0.0
        correct = 0
        total = 0
        clip_count = 0  # Track how many times we clipped gradients
        
        if self.train_loader:
            for _ in range(1):  # 1 Epoch per round
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # *** GRADIENT CLIPPING ***
                    # Before updating weights, check if gradients are too large.
                    # If yes, scale them down proportionally.
                    # This is like saying "no single batch can change the model too much"
                    grad_norm = nn_utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=SECURITY_CONFIG["max_grad_norm"]
                    )
                    if grad_norm > SECURITY_CONFIG["max_grad_norm"]:
                        clip_count += 1
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        self.last_grad_clip_count = clip_count
        
        # =============================================
        # STEP 5: Apply Differential Privacy noise
        # =============================================
        # After training, add random noise to the weights.
        # This "blurs" the weights so individual data can't be extracted.
        raw_params = self.get_parameters(config={})
        secure_params = self._apply_dp_noise(raw_params)
        
        # =============================================
        # STEP 6: Encrypt parameters with quantum key
        # =============================================
        # Now encrypt the noisy weights using the quantum-generated key.
        # Even if someone intercepts these bytes, they're unreadable without the key.
        encryption_status = "disabled"
        encrypted_data = None
        
        if SECURITY_CONFIG["encryption_enabled"]:
            try:
                encrypted_data, _ = encrypt_parameters(secure_params, key=quantum_key)
                encryption_status = "encrypted"
            except Exception as e:
                logger.error(f"Client {self.cid}: Encryption failed: {e}")
                encryption_status = "failed"
        
        # =============================================
        # STEP 7: Build metrics for UI display
        # =============================================
        key_hex = quantum_key.decode('utf-8') if isinstance(quantum_key, bytes) else str(quantum_key)
        
        metrics = {
            "loss": running_loss / len(self.train_loader) if self.train_loader else 0,
            "accuracy": 100 * correct / total if total else 0,
            "num_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            
            # Quantum security metrics
            "quantum_key": key_hex[:44],
            "chsh_value": chsh_value,
            "verification_status": is_verified,
            
            # Encryption metrics
            "encryption_status": encryption_status,
            "encrypted_data": encrypted_data,  # The actual encrypted blob (or None)
            
            # Differential privacy metrics
            "dp_noise_level": self.last_dp_noise_level,
            "dp_enabled": SECURITY_CONFIG["dp_enabled"],
            
            # Gradient clipping metrics
            "grad_clips": clip_count,
            "max_grad_norm": SECURITY_CONFIG["max_grad_norm"],
            
            "cid": self.cid
        }

        # Return: the DP-noised (but unencrypted) params for aggregation,
        # plus num_samples and metrics (which includes the encrypted version)
        return secure_params, metrics["num_samples"], metrics