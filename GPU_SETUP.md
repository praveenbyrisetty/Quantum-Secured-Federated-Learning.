# GPU Setup Guide for FLQC

## Current Status

Your system is configured to use GPU (`device: "cuda"` in config.yaml), but PyTorch CPU-only version is installed.

## Install CUDA-Enabled PyTorch

### Step 1: Check Your CUDA Version

Run this command to check if you have NVIDIA GPU and CUDA installed:

```bash
nvidia-smi
```

This will show your GPU model and CUDA version.

### Step 2: Uninstall CPU-only PyTorch

```bash
pip uninstall torch torchvision
```

### Step 3: Install CUDA-Enabled PyTorch

Choose based on your CUDA version:

#### For CUDA 11.8:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

#### For CUDA 12.4:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Verify GPU Installation

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see:

```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 (or your GPU model)
```

### Step 5: Restart Streamlit Server

After installing CUDA PyTorch:

1. Stop the current server (Ctrl+C)
2. Restart: `python -m streamlit run server.py`

The system will now use GPU automatically!

## Alternative: Use CPU (If No GPU Available)

If you don't have an NVIDIA GPU, edit `config.yaml`:

```yaml
system:
  device: "cpu" # Change from "cuda" to "cpu"
```

## Troubleshooting

### "CUDA out of memory" Error

Reduce batch sizes in `config.yaml`:

```yaml
training:
  batch_sizes:
    image: 16 # Reduced from 32
    text: 16
    tabular: 16
```

### GPU Not Detected After Installation

1. Ensure NVIDIA drivers are installed
2. Ensure CUDA toolkit is installed
3. Match PyTorch CUDA version with your system CUDA version

## Performance Comparison

**CPU Training (Current):**

- ~10-15 minutes for 5 rounds
- ~2-4 GB RAM

**GPU Training (After Fix):**

- ~2-5 minutes for 5 rounds
- ~1-2 GB VRAM
- **3-5x faster!**

## Quick Commands

```bash
# Check GPU
nvidia-smi

# Uninstall CPU PyTorch
pip uninstall torch torchvision -y

# Install GPU PyTorch (CUDA 11.8 - most common)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Restart server
python -m streamlit run server.py
```
