# FLQC: Quantum-Secured Federated Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-green.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

FLQC is a **quantum-secured federated learning framework** with a multi-layered security architecture. It demonstrates privacy-preserving machine learning across heterogeneous data distributions using E91 quantum key distribution (simulated), Byzantine-robust aggregation, and differential privacy.

### Key Features

- **Heterogeneous Data Distribution**: Three clients with different datasets:
  - 🚗 **Client 1**: CIFAR-10 non-living things (airplane, automobile, ship, truck)
  - 🐾 **Client 2**: CIFAR-10 living things (bird, cat, deer, dog, frog, horse)
  - 🔢 **Client 3**: MNIST handwritten digits (0-9, preprocessed to 32×32 RGB)

- **3-Layer Security Architecture**:
  - **Layer 1 — Communication**: E91 quantum key distribution + Fernet AES-128 encryption of model parameters + CHSH entanglement verification
  - **Layer 2 — Aggregation**: Krum + Trimmed Mean hybrid (default), with anomaly detection and norm clipping
  - **Layer 3 — Endpoint**: Gradient clipping + Differential Privacy noise injection

- **4 Aggregation Strategies** (selectable in sidebar):
  - **Krum + Trimmed Mean** (default, recommended): Krum scores and excludes suspicious clients, then Trimmed Mean aggregates the rest
  - **Trimmed Mean**: Removes extreme parameter values before averaging
  - **Krum**: Selects the single most trustworthy client update
  - **FedAvg**: Standard weighted average (baseline, no defense)

- **Interactive Streamlit UI**:
  - Real-time training visualization per client
  - Per-round security reports (anomaly detection, encryption status, CHSH values)
  - Security summary dashboard
  - Global model evaluation with confusion matrix
  - Image prediction interface

### ⚠️ Important: Quantum Simulation Disclosure

**This project uses SIMULATED quantum key distribution**, not real quantum hardware. The quantum components run on Qiskit's classical simulator (AerSimulator) for educational and demonstration purposes.

**What this means:**

- ✅ Demonstrates quantum-inspired concepts and workflows
- ✅ Uses quantum circuit simulation (Bell pairs) for key generation
- ✅ Implements CHSH inequality verification for eavesdropper detection
- ❌ Does NOT provide quantum-level security guarantees
- ❌ Does NOT use real quantum entanglement
- ❌ Does NOT protect against quantum computer attacks

**For production quantum security**, you would need:

- Real quantum hardware (IBM Quantum, IonQ, etc.)
- Post-quantum cryptography (NIST PQC standards)
- Proper quantum key distribution infrastructure

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Federated Server (server.py)                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Secure Aggregation: Krum + Trimmed Mean (Hybrid)          │  │
│  │  + Anomaly Detection + Norm Clipping                       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────┬──────────────────┬────────────────┘
           │ 🔐 Encrypted     │ 🔐 Encrypted     │ 🔐 Encrypted
   ┌───────▼──────┐   ┌───────▼──────┐    ┌──────▼───────┐
   │  Client 1    │   │  Client 2    │    │  Client 3    │
   │  CIFAR-10    │   │  CIFAR-10    │    │    MNIST     │
   │  Non-living  │   │   Living     │    │   Digits     │
   │              │   │              │    │              │
   │  CNN Model   │   │  CNN Model   │    │  CNN Model   │
   │  + Grad Clip │   │  + Grad Clip │    │  + Grad Clip │
   │  + DP Noise  │   │  + DP Noise  │    │  + DP Noise  │
   │  + E91 Enc   │   │  + E91 Enc   │    │  + E91 Enc   │
   └──────────────┘   └──────────────┘    └──────────────┘
```

### Security Flow Per Training Round

```
1. Server sends global model weights to each client
2. Each client:
   a. Generates quantum key via E91 protocol (simulated)
   b. Verifies entanglement via CHSH inequality test
   c. Trains locally with gradient clipping (limits data leakage)
   d. Adds Differential Privacy noise (prevents data reconstruction)
   e. Encrypts weights with quantum key (protects during transmission)
   f. Sends encrypted weights + key to server
3. Server:
   a. Decrypts all client updates
   b. Runs anomaly detection (flags suspicious norms)
   c. Applies Krum scoring (ranks clients by trustworthiness)
   d. Excludes least trustworthy client(s)
   e. Applies Trimmed Mean on remaining trusted updates
   f. Produces new global model
4. Repeat for N rounds
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/praveenbyrisetty/Quantum-Secured-Federated-Learning.git
   cd Quantum-Secured-Federated-Learning
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Quick Start

Run the federated learning system with Streamlit UI:

```bash
streamlit run server.py
```

Then:

1. Select an aggregation strategy in the sidebar (Krum + Trimmed Mean is default)
2. Set the number of training rounds
3. Click **"🚀 START FL TRAINING"**
4. Watch live training progress with security indicators per client
5. Review per-round security reports (anomaly detection, CHSH values)
6. View global model evaluation with confusion matrix
7. Test predictions by uploading images

---

## Project Structure

```
.
├── server.py                # Streamlit UI + FL orchestration + aggregation strategies
├── client_flwr.py           # FL client with gradient clipping, DP noise, encryption
├── quantum_e91.py           # E91 quantum key generation, CHSH verification, encryption
├── multi_modal_model.py     # CNN model architecture (CIFAR-10/MNIST)
├── data_setup.py            # Dataset loaders (CIFAR-10 partitioning + MNIST)
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── GPU_SETUP.md             # GPU setup instructions
├── README.md                # This file
└── data/                    # Data directory (auto-created on first run)
```

---

## Features in Detail

### Aggregation Strategies

| Strategy                | Security | How It Works                                                     | Best For                     |
| ----------------------- | -------- | ---------------------------------------------------------------- | ---------------------------- |
| **Krum + Trimmed Mean** | ⭐⭐⭐   | Krum excludes suspicious clients, Trimmed Mean averages the rest | Maximum security (default)   |
| **Trimmed Mean**        | ⭐⭐     | Sorts values, removes extremes, averages middle                  | Balanced security + accuracy |
| **Krum**                | ⭐⭐     | Selects single most trustworthy update                           | Byzantine fault tolerance    |
| **FedAvg**              | ❌       | Simple weighted average                                          | Trusted environments only    |

### Communication Security (Layer 1)

- **E91 Quantum Key Distribution**: Simulates entangled Bell pairs using Qiskit for key generation
- **CHSH Inequality Test**: Verifies entanglement integrity (S > 2.0 = no eavesdropper)
- **Fernet Encryption**: AES-128-CBC + HMAC for authenticated encryption of model parameters
- **Per-client Keys**: Each client generates a unique quantum key per round

### Endpoint Security (Layer 3)

- **Gradient Clipping**: Bounds gradient norm during local training (`max_grad_norm = 1.0`)
- **Differential Privacy Noise**: Gaussian noise injection after training (`noise_multiplier = 0.01`)
- **Security Metrics Reporting**: Each client reports clip count, DP noise level, and encryption status

### Anomaly Detection

- **Statistical Detection**: Flags clients whose update norm is > 2σ from the mean
- **Threshold Detection**: Rejects updates exceeding absolute norm threshold (1500.0)
- **Per-round Reports**: Visible in the UI after each aggregation round

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'flwr'
```

**Solution**: Make sure you're in the virtual environment: `venv\Scripts\activate`

### CUDA Out of Memory

**Solution**: The system automatically falls back to CPU if CUDA is unavailable. For explicit CPU mode, the device is auto-detected in `server.py`.

### Dataset Not Found

CIFAR-10 and MNIST are automatically downloaded on first run to the `data/` directory.

---

## Contributing

Contributions are welcome! Areas for improvement:

- Integration with real quantum hardware
- Additional aggregation algorithms (e.g., Median, Bulyan)
- More sophisticated privacy accounting (Rényi DP)
- Distributed deployment across multiple machines
- Additional model architectures

Fork the repo and submit a pull request!

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flqc2026,
  title={FLQC: Quantum-Secured Federated Learning},
  author={Praveen Byrisetty},
  year={2026},
  url={https://github.com/praveenbyrisetty/Quantum-Secured-Federated-Learning}
}
```

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Qiskit](https://qiskit.org/), [Flower](https://flower.ai/), and [Streamlit](https://streamlit.io/)
- Inspired by federated learning research and quantum cryptography
- CIFAR-10 dataset from [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- MNIST dataset from [Yann LeCun](http://yann.lecun.com/exdb/mnist/)
