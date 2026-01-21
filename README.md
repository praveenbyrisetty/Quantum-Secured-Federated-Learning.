# FLQC: Quantum-Enabled Federated Learning Architecture

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-green.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
FLQC is a proof-of-concept for a secure federated learning (FL) system enhanced with quantum cryptography. It trains a "universal" neural network model on heterogeneous datasets (vehicles/animals from CIFAR-10 and digits from MNIST) across three clients, without sharing raw data. Model updates are encrypted using simulated E91 quantum key distribution (via Qiskit) before aggregation on the server using FedAvg.

Key Features:
- **Heterogeneous Data Handling**: Splits CIFAR-10 and remaps MNIST for compatibility.
- **Quantum Security**: E91 protocol simulation for key generation and AES encryption of weights.
- **GPU Support**: Accelerated training with PyTorch CUDA.
- **Evaluation & Demo**: Multi-round training, accuracy plotting, and interactive inference demo.
- **Modular Design**: Separate files for model, data, clients, server, and quantum utils.

This project demonstrates privacy-preserving ML with quantum-inspired security, ideal for distributed systems in healthcare, finance, or IoT.

## Setup & Installation
1. Clone the repo: `git clone https://github.com/yourusername/flqc.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python server.py`

## Files
- `model.py`: Universal CNN model.
- `data_setup.py`: Heterogeneous data loaders.
- `quantum_e91.py`: E91 key generation and encryption.
- `client_*.py`: Client training scripts.
- `server.py`: Main FL orchestration.

## Usage
- Train: Run `server.py` for 5 rounds.
- Demo: Interactive prediction after training.
- GPU: Auto-detects CUDA if available.

## Contributing
Fork and PR! Issues welcome.

## License
MIT License.