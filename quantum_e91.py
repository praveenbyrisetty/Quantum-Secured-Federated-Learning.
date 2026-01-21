from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import hashlib
import base64
import pickle
from cryptography.fernet import Fernet

def generate_key():
    """
    Simulate E91: Entangle qubits, measure for random bits, hash to key.
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Superposition
    qc.cx(0, 1)  # Entangle
    qc.measure([0, 1], [0, 1])  # Measure

    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=256)  # More shots for better randomness
    result = job.result().get_counts()
    bit_string = ''.join([k * v for k, v in result.items()])
    digest = hashlib.sha256(bit_string.encode()).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_data(data_dict):
    key = generate_key()
    f = Fernet(key)
    serialized = pickle.dumps(data_dict)
    encrypted = f.encrypt(serialized)
    return encrypted, key

def decrypt_data(encrypted, key):
    f = Fernet(key)
    decrypted = f.decrypt(encrypted)
    return pickle.loads(decrypted)