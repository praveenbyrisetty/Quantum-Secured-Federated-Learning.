# FLQC: Quantum-Secured Federated Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2.4-green.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

FLQC is a **quantum-secured federated learning framework** with a multi-layered security architecture. It demonstrates privacy-preserving machine learning across heterogeneous data distributions using E91 quantum key distribution (simulated), Byzantine-robust aggregation, and differential privacy — applied to dermoscopic skin lesion classification using the **HAM10000** dataset.

### Key Features

- **Homogeneous Data Distribution** — IID (Independent and Identically Distributed) split across 3 simulated hospitals:
  - 🏥 **Hospital A (Client 0)**: All 7 classes
  - 🏥 **Hospital B (Client 1)**: All 7 classes
  - 🏥 **Hospital C (Client 2)**: All 7 classes

  > The full HAM10000 dataset is randomly shuffled and evenly distributed. This ensures models learn uniformly and prevents security aggregators (like Krum) from falsely flagging honest hospitals as anomalous due to data skew.

- **3-Layer Security Architecture**:
  - **Layer 1 — Communication**: E91 quantum key distribution + Fernet AES-128 encryption of model parameters + CHSH entanglement verification
  - **Layer 2 — Aggregation**: Krum + Trimmed Mean quantum e91 (default), with anomaly detection, cosine similarity checks, and norm clipping
  - **Layer 3 — Endpoint**: Gradient clipping + Differential Privacy noise injection (formal Gaussian mechanism with ε, δ guarantees)

- **4 Aggregation Strategies** (selectable in sidebar):
  - **Krum + Trimmed Mean** (default, recommended): Krum scores and excludes suspicious clients, then Trimmed Mean aggregates the rest
  - **Trimmed Mean**: Removes extreme parameter values before averaging
  - **Krum**: Selects the single most trustworthy client update
  - **FedAvg**: Standard weighted average (baseline, no defense)

- **Interactive Streamlit UI**:
  - Real-time training visualization per client
  - Per-round security reports (anomaly detection, encryption status, CHSH values)
  - Security summary dashboard
  - Global model evaluation with confusion matrix (all 7 classes)
  - Skin lesion prediction interface with medical guidance

### ⚠️ Important: Quantum Simulation Disclosure

**This project uses SIMULATED quantum key distribution**, not real quantum hardware. The quantum components run on Qiskit's classical simulator (AerSimulator) for educational and demonstration purposes.

**What this means:**

- ✅ Demonstrates quantum-inspired concepts and workflows
- ✅ Uses quantum circuit simulation (Bell pairs) for key generation
- ✅ Implements CHSH inequality verification for eavesdropper detection (S > 2.0 required)
- ✅ CHSH can genuinely fail (~5% chance due to shot noise), triggering a real transmission block
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
│              Federated Server (server.py)                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Secure Aggregation: Krum + Trimmed Mean (Quantum E91)          │  │
│  │  + Norm Clipping + Anomaly Detection (norm + cosine sim)   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────┬──────────────────┬────────────────┘
           │ 🔐 Encrypted     │ 🔐 Encrypted     │ 🔐 Encrypted
   ┌───────▼──────┐   ┌───────▼──────┐    ┌──────▼───────┐
   │  Hospital A  │   │  Hospital B  │    │  Hospital C  │
   │All 7 classes │   │All 7 classes │    │All 7 classes │
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
   a. Generates quantum key via E91 protocol (simulated Qiskit Bell pairs)
   b. Verifies entanglement via CHSH inequality test (S > 2.0 required)
      → If S ≤ 2.0: transmission BLOCKED, client skipped this round
   c. Trains locally with gradient clipping (bounds DP sensitivity Δf)
   d. Adds Differential Privacy noise via Gaussian mechanism (σ = Δf·√(2ln(1.25/δ))/ε)
   e. Writes DP-noised weights back onto model (so get_parameters() always returns secured weights)
   f. Encrypts weights with quantum key + HMAC integrity tag
   g. Sends DP-noised weights + encrypted blob to server
3. Server:
   a. Filters out CHSH-blocked clients
   b. Runs anomaly detection (norm + cosine similarity)
   c. Applies norm clipping (caps each update at NORM_THRESHOLD)
   d. Applies Krum scoring (ranks clients by trustworthiness)
   e. Excludes least trustworthy client(s)
   f. Applies Trimmed Mean on remaining trusted updates
   g. Produces new global model
4. Repeat for N rounds
```

---

## Dataset: HAM10000

The HAM10000 (Human Against Machine with 10,000 training images) dataset contains **10,015 dermoscopic images** across **7 skin lesion classes**:

| Class   | Full Name            | Type          |
| ------- | -------------------- | ------------- |
| `akiec` | Actinic Keratoses    | Pre-malignant |
| `bcc`   | Basal Cell Carcinoma | Malignant     |
| `bkl`   | Benign Keratosis     | Benign        |
| `df`    | Dermatofibroma       | Benign        |
| `mel`   | Melanoma             | ⚠️ Malignant  |
| `nv`    | Melanocytic Nevi     | Benign        |
| `vasc`  | Vascular Lesions     | Benign        |

Download: [Kaggle — HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

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

4. **Download the HAM10000 dataset**

   ```bash
   python download_dataset.py
   ```

   This auto-downloads from Kaggle (requires a free Kaggle account) and places the dataset at `./data/HAM10000/`. Alternatively, download manually and place the folder there.

---

## Usage

### Quick Start

```bash
streamlit run server.py
```

Then:

1. Select an aggregation strategy in the sidebar (Krum + Trimmed Mean is default)
2. Set the number of training rounds (1–10)
3. Click **"🚀 START TRAINING"**
4. Watch live training progress with security indicators per hospital
5. Review per-round security reports (anomaly detection, CHSH values, DP budget)
6. View global model evaluation with confusion matrix across all 7 classes
7. Upload a dermoscopic image for skin lesion prediction with medical guidance

---

## Project Structure

```
.
├── server.py              # Streamlit UI + FL orchestration + aggregation strategies
├── client_flwr.py         # FL client: gradient clipping, DP noise, E91 encryption
├── quantum_e91.py         # E91 quantum key generation, CHSH verification, encryption
├── quantum_e91_model.py        # CNN model (3 conv blocks → 3 FC layers, 7-class output)
├── data_setup.py          # HAM10000 dataset loader + non-IID hospital partitioning
├── download_dataset.py    # Kaggle auto-downloader for HAM10000
├── GPU_SETUP.md           # CUDA/GPU setup instructions
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/                  # Created on first run
    ├── HAM10000/          # Raw downloaded dataset
    └── HAM10000_organized/  # Organized by class folder (auto-created)
```

---

## Features in Detail

### Aggregation Strategies

| Strategy                | Security | How It Works                                                       | Best For                     |
| ----------------------- | -------- | ------------------------------------------------------------------ | ---------------------------- |
| **Krum + Trimmed Mean** | ⭐⭐⭐   | Krum excludes suspicious clients, Trimmed Mean aggregates the rest | Maximum security (default)   |
| **Trimmed Mean**        | ⭐⭐     | Sorts values per weight, removes extremes, averages the middle     | Balanced security + accuracy |
| **Krum**                | ⭐⭐     | Selects single most trustworthy update by pairwise distance        | Byzantine fault tolerance    |
| **FedAvg**              | ❌       | Simple weighted average by sample count                            | Trusted environments only    |

### Communication Security (Layer 1)

- **E91 Quantum Key Distribution**: Simulates entangled Bell pairs using Qiskit for per-round key generation
- **CHSH Inequality Test**: Verifies entanglement integrity using real Qiskit circuits across 4 basis combinations; S > 2.0 required — transmission is **blocked** if it fails
- **Fernet Encryption**: AES-128-CBC + HMAC-SHA256 integrity tag for authenticated encryption of model parameters
- **No Key Co-transmission**: The encryption key is never sent alongside the encrypted data

### Aggregation Security (Layer 2)

- **Norm Clipping**: Caps each client update norm at `NORM_THRESHOLD` before aggregation, limiting any single client's influence
- **Anomaly Detection**: Flags clients by statistical norm deviation (>2σ), absolute threshold, and pairwise cosine similarity (<0.5)
- **Krum Scoring**: Ranks clients by L2 distance to neighbours; least trustworthy are excluded
- **Trimmed Mean**: Per-weight trimming removes extreme values; minimum trim of 1 enforced even with 3 clients

### Endpoint Security (Layer 3)

- **Gradient Clipping**: Bounds gradient norm during local training at `max_grad_norm = 1.0`, which also defines the DP sensitivity Δf
- **Differential Privacy (Gaussian Mechanism)**: Formal (ε, δ)-DP guarantee per round using σ = Δf · √(2 ln(1.25/δ)) / ε
  - Default: ε = 5.0, δ = 1e-5 per round
  - Cumulative ε tracked and displayed in the UI across all rounds
- **DP applied before `get_parameters()`**: DP-noised weights are written back to the model so all downstream access is secured

### Safe Serialization

Model parameters are serialized using a custom numpy-based binary format (no `pickle`), preventing arbitrary code execution from a crafted malicious payload. HMAC-SHA256 integrity verification detects any tampering before decryption.

---

## Troubleshooting

### Dataset Not Found

```
FileNotFoundError: HAM10000 dataset not found
```

Run `python download_dataset.py` or manually download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place files in `./data/HAM10000/`.

### Import Errors

```
ModuleNotFoundError: No module named 'flwr'
```

Ensure you're in the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows).

### CUDA Out of Memory

Reduce batch sizes in `client_flwr.py` and `server.py` (default is 32). The system automatically falls back to CPU if CUDA is unavailable.

### CHSH Verification Failing Frequently

A ~5% failure rate per client per round is expected due to quantum shot noise with 1024 shots. If failures exceed ~15%, check that Qiskit Aer is properly installed: `pip install qiskit-aer`.

---

## Contributing

Contributions are welcome! Areas for improvement:

- Integration with real quantum hardware (IBM Quantum, IonQ)
- Advanced privacy accounting (Rényi DP, moments accountant)
- Additional aggregation algorithms (Median, Bulyan, FLTrust)
- Distributed deployment across multiple machines
- Stronger CNN backbone (ResNet, EfficientNet) for HAM10000

Fork the repo and submit a pull request!

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flqc2026,
  title  = {FLQC: Quantum-Secured Federated Learning for Skin Lesion Classification},
  author = {Praveen Byrisetty},
  year   = {2026},
  url    = {https://github.com/praveenbyrisetty/Quantum-Secured-Federated-Learning}
}
```

---

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Qiskit](https://qiskit.org/), [Flower](https://flower.ai/), and [Streamlit](https://streamlit.io/)
- HAM10000 dataset: Tschandl, P. et al. (2018). _The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions._ Scientific Data, 5, 180161.

.\venv\Scripts\activate; streamlit run server.py
