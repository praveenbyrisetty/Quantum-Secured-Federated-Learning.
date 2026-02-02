"""
Quantum-Inspired Encryption Module for FLQC

IMPORTANT: This module uses SIMULATED quantum key distribution via Qiskit's
classical simulator. This is NOT real quantum cryptography and does NOT provide
quantum-level security guarantees.

Real quantum key distribution requires:
- Actual quantum hardware (IBM Quantum, IonQ, etc.)
- Quantum channels for entanglement distribution
- Proper basis reconciliation and privacy amplification
- Authentication protocols

This implementation demonstrates the CONCEPT of E91 protocol for educational
purposes and provides quantum-inspired randomness for key generation.

For production systems requiring quantum security, use:
- Real quantum hardware
- Post-quantum cryptography (e.g., NIST PQC standards)
- Hybrid classical-quantum schemes
"""
import hashlib
import base64
import logging
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from cryptography.fernet import Fernet, InvalidToken
import pickle
import numpy as np
from typing import Tuple, Any, Optional

# Setup logging
logger = logging.getLogger("FLQC.Quantum")


# ==========================================
# QUANTUM KEY GENERATION (SIMULATED E91)
# ==========================================

def generate_key(length: int = 128) -> bytes:
    """
    Simulates E91 Quantum Key Distribution using entangled qubits.
    
    **SIMULATION NOTICE**: This uses Qiskit's classical simulator (AerSimulator),
    not real quantum hardware. The security is based on classical cryptographic
    hashing, not quantum mechanics.
    
    E91 Protocol Overview:
    1. Create entangled Bell pairs (|Φ+⟩ state)
    2. Alice and Bob measure in random bases
    3. Compare bases and keep matching results
    4. Use correlated bits as shared secret key
    
    Args:
        length: Number of key bits to generate (default: 128)
        
    Returns:
        Fernet-compatible encryption key (32 bytes, base64 encoded)
        
    Raises:
        RuntimeError: If quantum simulation fails
    """
    try:
        key_bits = []
        sim = AerSimulator()
        
        logger.debug(f"Generating {length}-bit quantum key...")
        
        # Generate 'length' bits using quantum circuit
        for i in range(length):
            qc = QuantumCircuit(2, 2)
            
            # Step 1: Create Bell State (Entanglement)
            # |00⟩ + |11⟩ (maximally entangled)
            qc.h(0)           # Hadamard on qubit 0 (superposition)
            qc.cx(0, 1)       # CNOT to entangle qubit 0 and 1
            
            # Step 2: Measure both qubits
            # In real E91, Alice and Bob would measure in different bases
            # and perform basis reconciliation
            qc.measure([0, 1], [0, 1])
            
            # Step 3: Simulate quantum measurement
            job = sim.run(transpile(qc, sim), shots=1, memory=True)
            result = job.result().get_memory()[0]
            
            # Use Alice's measurement (first qubit) as key bit
            # In real E91, correlated measurements provide security
            key_bits.append(result[0])
        
        # Convert bit string to cryptographic key
        key_string = "".join(key_bits)
        
        # Hash to ensure uniform distribution and proper length
        sha = hashlib.sha256(key_string.encode()).digest()
        
        # Fernet requires 32-byte base64-encoded key
        fernet_key = base64.urlsafe_b64encode(sha[:32])
        
        logger.debug(f"✓ Generated {length}-bit quantum-inspired key")
        return fernet_key
    
    except Exception as e:
        logger.error(f"Quantum key generation failed: {e}")
        raise RuntimeError(f"Failed to generate quantum key: {e}")


# ==========================================
# QUANTUM TRUE RANDOMNESS (For DP Seeding)
# ==========================================

def get_quantum_seed(bits: int = 32) -> int:
    """
    Generates a random integer using quantum circuit measurements.
    
    **SIMULATION NOTICE**: Uses simulated quantum randomness. On real quantum
    hardware, this would provide true physical randomness from quantum mechanics.
    
    Used to seed differential privacy noise generators with high-quality randomness.
    
    Args:
        bits: Number of bits for the random integer (default: 32)
        
    Returns:
        Random integer in range [0, 2^bits - 1]
        
    Raises:
        RuntimeError: If quantum simulation fails
    """
    try:
        qc = QuantumCircuit(1, 1)
        qc.h(0)  # Superposition: |0⟩ + |1⟩
        qc.measure(0, 0)
        
        sim = AerSimulator()
        
        # Run multiple shots to get 'bits' random bits
        job = sim.run(transpile(qc, sim), shots=bits, memory=True)
        memory = job.result().get_memory()
        
        # Convert bitstring to integer
        binary_string = "".join(memory)
        random_int = int(binary_string, 2)
        
        logger.debug(f"Generated {bits}-bit quantum seed: {random_int}")
        return random_int
    
    except Exception as e:
        logger.error(f"Quantum seed generation failed: {e}")
        # Fallback to classical randomness
        logger.warning("Falling back to classical random seed")
        return np.random.randint(0, 2**bits)


# ==========================================
# ENCRYPTION WRAPPERS
# ==========================================

def encrypt_data(data: Any) -> Tuple[bytes, bytes]:
    """
    Encrypt data using quantum-generated key and Fernet symmetric encryption.
    
    Process:
    1. Generate quantum-inspired encryption key
    2. Serialize data using pickle
    3. Encrypt with Fernet (AES-128 in CBC mode)
    
    Args:
        data: Any Python object (typically model weights dictionary)
        
    Returns:
        Tuple of (encrypted_data, encryption_key)
        
    Raises:
        RuntimeError: If encryption fails
        TypeError: If data cannot be serialized
    """
    try:
        # Generate quantum key
        key = generate_key()
        
        # Create Fernet cipher
        f = Fernet(key)
        
        # Serialize data
        try:
            serialized = pickle.dumps(data)
        except Exception as e:
            raise TypeError(f"Cannot serialize data: {e}")
        
        # Encrypt
        encrypted = f.encrypt(serialized)
        
        logger.debug(f"✓ Encrypted {len(serialized)} bytes → {len(encrypted)} bytes")
        return encrypted, key
    
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise RuntimeError(f"Failed to encrypt data: {e}")


def decrypt_data(encrypted_data: bytes, key: bytes) -> Any:
    """
    Decrypt data using provided key.
    
    Args:
        encrypted_data: Encrypted bytes from encrypt_data()
        key: Encryption key from encrypt_data()
        
    Returns:
        Decrypted and deserialized data
        
    Raises:
        RuntimeError: If decryption fails
        InvalidToken: If key is incorrect or data is corrupted
    """
    try:
        # Create Fernet cipher with provided key
        f = Fernet(key)
        
        # Decrypt
        try:
            decrypted = f.decrypt(encrypted_data)
        except InvalidToken:
            raise InvalidToken("Decryption failed: Invalid key or corrupted data")
        
        # Deserialize
        data = pickle.loads(decrypted)
        
        logger.debug(f"✓ Decrypted {len(encrypted_data)} bytes → {len(decrypted)} bytes")
        return data
    
    except InvalidToken:
        logger.error("Decryption failed: Invalid key or corrupted data")
        raise
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise RuntimeError(f"Failed to decrypt data: {e}")


# ==========================================
# VALIDATION UTILITIES
# ==========================================

def validate_encryption_key(key: bytes) -> bool:
    """
    Validate that a key is properly formatted for Fernet.
    
    Args:
        key: Encryption key to validate
        
    Returns:
        True if key is valid, False otherwise
    """
    try:
        # Try to create Fernet instance
        Fernet(key)
        return True
    except Exception:
        return False


def test_encryption_roundtrip(test_data: Any = None) -> bool:
    """
    Test encryption/decryption roundtrip to verify functionality.
    
    Args:
        test_data: Optional test data (default: simple dictionary)
        
    Returns:
        True if roundtrip successful, False otherwise
    """
    if test_data is None:
        test_data = {"test": "data", "number": 42}
    
    try:
        # Encrypt
        encrypted, key = encrypt_data(test_data)
        
        # Decrypt
        decrypted = decrypt_data(encrypted, key)
        
        # Verify
        if decrypted == test_data:
            logger.info("✓ Encryption roundtrip test passed")
            return True
        else:
            logger.error("✗ Encryption roundtrip test failed: data mismatch")
            return False
    
    except Exception as e:
        logger.error(f"✗ Encryption roundtrip test failed: {e}")
        return False


if __name__ == "__main__":
    # Run self-tests
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("FLQC Quantum Encryption Module - Self Test")
    print("=" * 50)
    
    print("\n1. Testing quantum key generation...")
    key = generate_key(128)
    print(f"   ✓ Generated key: {key[:20]}...")
    
    print("\n2. Testing quantum seed generation...")
    seed = get_quantum_seed(32)
    print(f"   ✓ Generated seed: {seed}")
    
    print("\n3. Testing encryption roundtrip...")
    test_encryption_roundtrip()
    
    print("\n4. Testing with model weights...")
    import torch
    fake_weights = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10)
    }
    encrypted, key = encrypt_data(fake_weights)
    decrypted = decrypt_data(encrypted, key)
    
    match = all(torch.equal(fake_weights[k], decrypted[k]) for k in fake_weights)
    if match:
        print("   ✓ Model weights encryption successful")
    else:
        print("   ✗ Model weights encryption failed")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
