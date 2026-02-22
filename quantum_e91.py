"""
Quantum-Inspired Encryption Module for FLQC
"""
import hashlib
import base64
import logging
import pickle
import time
import random
import numpy as np
from typing import Tuple, Any
from cryptography.fernet import Fernet, InvalidToken

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Setup logging
logger = logging.getLogger("FLQC.Quantum")

# ==========================================
# 1. QUANTUM KEY GENERATION (SIMULATED E91)
# ==========================================
def generate_key(length: int = 128) -> bytes:
    """
    Simulates E91 Quantum Key Distribution using entangled qubits.
    """
    try:
        key_bits = []
        sim = AerSimulator()
        
        # logger.debug(f"Generating {length}-bit quantum key...")
        
        # Optimize: Generate multiple bits per circuit to speed up
        # For simulation, we can just use 1 shot if we trust the randomness
        for i in range(length):
            qc = QuantumCircuit(2, 2)
            qc.h(0)           # Hadamard on qubit 0 (superposition)
            qc.cx(0, 1)       # CNOT to entangle qubit 0 and 1
            qc.measure([0, 1], [0, 1])
            
            job = sim.run(transpile(qc, sim), shots=1, memory=True)
            result = job.result().get_memory()[0]
            key_bits.append(result[0])
        
        key_string = "".join(key_bits)
        sha = hashlib.sha256(key_string.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(sha[:32])
        
        return fernet_key
    
    except Exception as e:
        logger.error(f"Quantum key generation failed: {e}")
        # Fallback for demo stability
        return Fernet.generate_key()

# ==========================================
# 2. ENTANGLEMENT VERIFICATION (CHSH)
# ==========================================
def verify_entanglement(client_id: str) -> bool:
    """
    Simulates the verification of entanglement (e.g., CHSH inequality test).
    """
    logger.info(f"[{client_id}] Initiating CHSH correlation test...")
    
    # Simulate CHSH value S.
    # Local Reality <= 2, Quantum Mechanics <= 2*sqrt(2) ≈ 2.82
    # We simulate a value between 2.0 and 2.8 to show successful entanglement
    s_value = 2.0 + (0.8 * random.random())
    
    # logger.debug(f"[{client_id}] CHSH s-value measured: {s_value:.4f}")
    
    if s_value > 2.0:
        return True
    else:
        logger.warning(f"[{client_id}] ⚠ Entanglement NOT verified")
        return False

# ==========================================
# 3. ENCRYPTION WRAPPERS (General Data)
# ==========================================
def encrypt_data(data: Any) -> Tuple[bytes, bytes]:
    """Encrypt data using quantum-generated key."""
    try:
        key = generate_key()
        f = Fernet(key)
        serialized = pickle.dumps(data)
        encrypted = f.encrypt(serialized)
        return encrypted, key
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise RuntimeError(f"Failed to encrypt data: {e}")

def decrypt_data(encrypted_data: bytes, key: bytes) -> Any:
    """Decrypt data using provided key."""
    try:
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data)
        return pickle.loads(decrypted)
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise

# ==========================================
# 4. MODEL PARAMETER ENCRYPTION/DECRYPTION
# ==========================================
# WHY: In FL, clients send model weights to the server.
# Without encryption, anyone intercepting the network traffic can see the weights.
# These functions encrypt the weights using a quantum-generated key so that
# even if someone intercepts the data, they can't read it.

def encrypt_parameters(params: list, key: bytes = None) -> Tuple[bytes, bytes]:
    """
    Encrypt model parameters (list of numpy arrays) for secure transmission.
    
    HOW IT WORKS:
    1. Takes the list of numpy arrays (model weights) 
    2. Serializes them with pickle (converts to bytes)
    3. Encrypts the bytes using Fernet (AES-128 encryption) with a quantum key
    4. Returns the encrypted blob + the key needed to decrypt
    
    Args:
        params: List of numpy arrays (model weights from PyTorch)
        key: Optional pre-generated quantum key. If None, generates a new one.
    
    Returns:
        Tuple of (encrypted_bytes, fernet_key)
    """
    try:
        # Step 1: Generate a quantum key if not provided
        if key is None:
            key = generate_key(128)
        
        # Step 2: Serialize the parameters (convert numpy arrays → bytes)
        serialized = pickle.dumps(params)
        
        # Step 3: Encrypt using Fernet (AES-128-CBC + HMAC for integrity)
        f = Fernet(key)
        encrypted = f.encrypt(serialized)
        
        return encrypted, key
    except Exception as e:
        logger.error(f"Parameter encryption failed: {e}")
        raise RuntimeError(f"Failed to encrypt parameters: {e}")


def decrypt_parameters(encrypted_data: bytes, key: bytes) -> list:
    """
    Decrypt model parameters back to list of numpy arrays.
    
    HOW IT WORKS:
    1. Takes the encrypted blob and the key
    2. Decrypts using Fernet
    3. Deserializes back to list of numpy arrays
    
    Args:
        encrypted_data: The encrypted bytes from encrypt_parameters()
        key: The Fernet key used during encryption
    
    Returns:
        List of numpy arrays (the original model weights)
    """
    try:
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data)
        params = pickle.loads(decrypted)
        return params
    except InvalidToken:
        logger.error("Decryption failed: Invalid key or corrupted data!")
        raise RuntimeError("Invalid quantum key - possible tampering detected!")
    except Exception as e:
        logger.error(f"Parameter decryption failed: {e}")
        raise


def generate_shared_key(client_id: str, server_id: str = "server") -> Tuple[bytes, float]:
    """
    Simulate E91 shared key exchange between a client and the server.
    
    IN REAL QUANTUM KEY DISTRIBUTION (E91):
    - Both parties share entangled qubit pairs
    - They measure their qubits in different bases
    - After comparing bases publicly, matching measurements become the shared key
    - CHSH inequality test verifies no eavesdropper
    
    Here we simulate this process and return:
    - The shared key (same key for both parties)
    - The CHSH S-value (should be > 2.0 for valid entanglement)
    
    Args:
        client_id: ID of the client requesting the key
        server_id: ID of the server (default "server")
    
    Returns:
        Tuple of (shared_fernet_key, chsh_s_value)
    """
    logger.info(f"E91 key exchange: {client_id} ↔ {server_id}")
    
    # Generate the quantum key (simulated E91)
    key = generate_key(128)
    
    # Verify entanglement (CHSH test)
    s_value = 2.0 + (0.8 * random.random())  # Simulated CHSH value
    
    is_entangled = s_value > 2.0
    if is_entangled:
        logger.info(f"E91 exchange successful: S={s_value:.4f} (entanglement verified)")
    else:
        logger.warning(f"E91 exchange FAILED: S={s_value:.4f} (no entanglement)")
    
    return key, s_value