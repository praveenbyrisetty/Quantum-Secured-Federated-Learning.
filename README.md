# FLQC: Quantum-Inspired Federated Learning Architecture

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-green.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

FLQC is a **multi-modal federated learning framework** with quantum-inspired security features. It demonstrates privacy-preserving machine learning across three different data modalities (images, text, and tabular data) using differential privacy, Byzantine-robust aggregation, and quantum-inspired encryption.

### Key Features

- **Multi-Modal Architecture**: Trains three separate models simultaneously:
  - 🚗 **Image Model** (CNN): Vehicle classification from CIFAR-10
  - 📝 **Text Model** (LSTM): Security log analysis
  - 📊 **Tabular Model** (MLP): IoT sensor anomaly detection

- **Privacy Protection**:
  - Differential Privacy with configurable ε/δ budgets
  - Gradient clipping and calibrated Gaussian noise
  - Quantum-seeded randomness for noise generation

- **Security Features**:
  - Quantum-inspired E91 key distribution (simulated)
  - AES-128 encryption via Fernet
  - Byzantine-robust aggregation (FedAvg, Krum, Trimmed Mean)
  - Poisoning attack detection via norm thresholds

- **Production-Ready**:
  - YAML-based configuration system
  - Comprehensive logging and error handling
  - Modular, well-documented codebase
  - Interactive Streamlit UI

### ⚠️ Important: Quantum Simulation Disclosure

**This project uses SIMULATED quantum key distribution**, not real quantum hardware. The quantum components run on Qiskit's classical simulator (AerSimulator) for educational and demonstration purposes.

**What this means:**

- ✅ Demonstrates quantum-inspired concepts and workflows
- ✅ Uses quantum circuit simulation for key generation
- ✅ Provides high-quality pseudorandom number generation
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
┌─────────────────────────────────────────────────────────┐
│                    Federated Server                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Secure Aggregation (FedAvg/Krum/Trimmed Mean)   │   │
│  │  + Poisoning Detection + Privacy Budget Tracking │   │
│  └──────────────────────────────────────────────────┘   │
└───────────┬──────────────┬──────────────┬───────────────┘
            │              │              │
    ┌───────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
    │  Client 1    │ │ Client 2 │ │  Client 3   │
    │  (Images)    │ │  (Text)  │ │  (Tabular)  │
    │              │ │          │ │             │
    │  CNN Model   │ │   LSTM   │ │  MLP Model  │
    │  + DP Noise  │ │ + DP     │ │  + DP Noise │
    │  + E91 Enc   │ │ + E91    │ │  + E91 Enc  │
    └──────────────┘ └──────────┘ └─────────────┘
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

4. **Configure the system** (optional)

   Edit `config.yaml` to customize:
   - Training parameters (learning rates, epochs, batch sizes)
   - Privacy budgets (epsilon, delta)
   - Security settings (aggregation method, thresholds)
   - Data paths

---

## Usage

### Quick Start

Run the federated learning system with Streamlit UI:

```bash
streamlit run server.py
```

Then:

1. Click "🚀 INITIALIZE TRAINING" in the sidebar
2. Watch the training progress across 3 clients
3. Test the trained models in the interactive tabs

### Configuration

The system is configured via `config.yaml`. Key parameters:

```yaml
training:
  num_rounds: 5 # Number of federated rounds
  learning_rates:
    image: 0.01 # CNN learning rate
    text: 0.001 # LSTM learning rate
    tabular: 0.001 # MLP learning rate

privacy:
  enabled: true
  epsilon: 1.0 # Privacy budget (lower = more private)
  delta: 1e-5
  max_grad_norm: 1.0 # Gradient clipping threshold

security:
  aggregation_method: "fedavg" # Options: fedavg, krum, trimmed_mean
  norm_threshold: 1500.0 # Poisoning detection threshold
```

### Running Tests

```bash
# Test encryption module
python quantum_e91.py

# Test configuration loading
python config_loader.py

# Check imports
python -c "from client_1 import run_client_1; from client_2 import run_client_2; from client_3 import run_client_3; print('✓ All imports successful')"
```

---

## Project Structure

```
.
├── config.yaml              # Configuration file
├── config_loader.py         # Configuration management
├── constants.py             # Named constants
├── utils.py                 # Utility functions (aggregation, DP, metrics)
├── quantum_e91.py           # Quantum-inspired encryption
├── multi_modal_model.py     # Neural network models
├── data_setup.py            # Dataset loaders
├── client_1.py              # Image classification client
├── client_2.py              # Text classification client
├── client_3.py              # Tabular classification client
├── server.py                # Streamlit UI + orchestration
├── requirements.txt         # Python dependencies
└── data/                    # Data directory
    ├── images/              # Custom images (optional)
    ├── test.txt             # Text data (optional)
    └── table.csv            # Tabular data (optional)
```

---

## Features in Detail

### Differential Privacy

- **Gradient Clipping**: Bounds sensitivity before adding noise
- **Calibrated Noise**: Gaussian noise scaled by privacy parameters
- **Privacy Budget Tracking**: Monitors cumulative ε spent across rounds
- **Quantum Seeding**: Uses quantum circuits for high-quality randomness

### Byzantine-Robust Aggregation

- **FedAvg**: Standard averaging (default)
- **Krum**: Selects most trustworthy update based on distance to neighbors
- **Trimmed Mean**: Removes extreme values before averaging

### Quantum-Inspired Security

- **E91 Protocol Simulation**: Creates entangled Bell pairs for key generation
- **Fernet Encryption**: AES-128 in CBC mode with HMAC
- **Key Validation**: Ensures encryption integrity

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'yaml'
```

**Solution**: Install PyYAML: `pip install PyYAML`

### CUDA Out of Memory

**Solution**: Reduce batch sizes in `config.yaml` or use CPU:

```yaml
system:
  device: "cpu"
```

### Dataset Not Found

The system automatically falls back to CIFAR-10 and synthetic data if local files are missing. To use custom data:

- Place images in `data/images/` with subdirectories for each class
- Add text logs to `data/test.txt`
- Add CSV with features in `data/table.csv`

---

## Contributing

Contributions are welcome! Areas for improvement:

- Integration with real quantum hardware
- Additional aggregation algorithms
- More sophisticated privacy accounting
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
  title={FLQC: Quantum-Inspired Federated Learning Architecture},
  author={Praveen Byrisetty},
  year={2026},
  url={https://github.com/praveenbyrisetty/Quantum-Secured-Federated-Learning}
}
```

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Qiskit](https://qiskit.org/), and [Streamlit](https://streamlit.io/)
- Inspired by federated learning research and quantum cryptography
- CIFAR-10 dataset from [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
