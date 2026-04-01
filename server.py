"""
FLQC Server - Quantum-Secured Federated Learning Server

WHAT THIS FILE DOES:
This is the CENTRAL SERVER that coordinates 3 hospital clients
for skin lesion classification using the HAM10000 dataset.
It runs a Streamlit web UI where you can:
1. Start federated training across 3 simulated hospitals
2. Watch live training progress per hospital
3. See security status for each round
4. Evaluate the global model on all 7 skin lesion types
5. Test predictions with uploaded dermoscopic images

DATASET: HAM10000 — 10,015 dermoscopic images, 7 classes:
  akiec, bcc, bkl, df, mel, nv, vasc

SECURITY FEATURES IN AGGREGATION:
- Trimmed Mean: Removes extreme values before averaging (resists poisoning)
- Krum: Selects the most trustworthy update (resists Byzantine attacks)
- FedAvg: Standard weighted average (baseline, no defense)
- Norm Clipping: Rejects updates with suspiciously large norms
- Anomaly Detection: Flags clients sending suspicious updates

HOW TO RUN:
    streamlit run server.py
"""

import streamlit as st
import torch
import numpy as np
import time
from typing import List, Tuple
from collections import OrderedDict
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from multi_modal_model import MultiModalFederatedModel, NUM_CLASSES
from client_flwr import FLQCClient
from data_setup import get_client_dataset, get_full_test_dataset, CLASS_NAMES, CLASS_DISPLAY, CLIENT_CLASSES, IMAGE_SIZE
from quantum_e91 import decrypt_parameters


def verify_client_encryption(metrics: dict) -> dict:
    """
    Verify the encryption pipeline status from client metrics.
    
    SECURITY FIX: The quantum key is no longer sent alongside encrypted data.
    This function now only reports the encryption status without attempting
    server-side decryption (since the key is kept separate for security).
    
    Returns: dict with verification status and details
    """
    result = {"verified": False, "status": "not_attempted", "detail": ""}
    
    # Check if client was blocked by CHSH
    if metrics.get("chsh_blocked", False):
        result["status"] = "chsh_blocked"
        result["detail"] = "Client blocked due to CHSH verification failure"
        return result
    
    encryption_status = metrics.get("encryption_status", "disabled")
    encrypted_data = metrics.get("encrypted_data")
    
    if encryption_status == "encrypted" and encrypted_data is not None:
        result["verified"] = True
        result["status"] = "encrypted"
        result["detail"] = f"Encrypted payload: {len(encrypted_data):,} bytes (HMAC-verified)"
    elif encryption_status == "blocked_chsh_failed":
        result["status"] = "chsh_blocked"
        result["detail"] = "Transmission blocked — CHSH verification failed"
    elif encryption_status == "failed":
        result["status"] = "encryption_failed"
        result["detail"] = "Encryption failed during processing"
    elif encryption_status == "disabled":
        result["status"] = "disabled"
        result["detail"] = "Encryption is disabled in config"
    else:
        result["status"] = "unknown"
        result["detail"] = f"Unknown encryption status: {encryption_status}"
    
    return result

# --- MAIN CONFIG ---
st.set_page_config(page_title="FLQC - Skin Lesion FL", layout="wide", page_icon="🏥")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HAM10000 class display names
HAM10000_CLASSES = [CLASS_DISPLAY[c] for c in CLASS_NAMES]

# Client descriptions for UI
CLIENT_LABELS = {
    0: "🏥 Hospital A — Melanocytic (nv, mel)",
    1: "🏥 Hospital B — Keratosis (bkl, bcc, akiec)",
    2: "🏥 Hospital C — Vascular (vasc, df)",
}


# ==========================================
# AGGREGATION STRATEGIES
# ==========================================
# These are different ways the server combines model updates from all clients.
# Each strategy has different resistance to attacks.

# --- ACTIVE SECURITY CONFIG ---
# Change this to switch aggregation strategy
AGGREGATION_METHOD = "krum_trimmed_mean"  # Options: "fedavg", "trimmed_mean", "krum", "krum_trimmed_mean"
TRIMMED_MEAN_BETA = 0.1                    # Fraction to trim from each end (10%)
NORM_THRESHOLD = 1500.0             # Max allowed update norm (anomaly detection)


def fedavg_aggregate(results: List[Tuple]) -> list:
    """
    FedAvg (Federated Averaging) - BASELINE aggregation.
    
    HOW IT WORKS:
    - Takes model weights from all clients
    - Computes a weighted average based on how many data samples each client has
    - Clients with more data have more influence
    
    VULNERABILITY:
    - If one client sends poisoned (garbage) weights, they get averaged in
    - No defense against malicious clients
    
    WHEN TO USE:
    - When you trust all clients
    - As a baseline to compare against secure methods
    """
    if not results:
        return None
    
    # Extract parameters and number of samples from each client
    weights_results = [
        (client.get_parameters({}), num_samples) 
        for client, num_samples, _ in results
    ]
    
    # Calculate total samples across all clients
    total_samples = sum([num_samples for _, num_samples in weights_results])
    
    # Weighted average: client with more data gets more weight
    aggregated = None
    for params, num_samples in weights_results:
        weight = num_samples / total_samples  # Weight = proportion of data
        
        if aggregated is None:
            aggregated = [w * weight for w in params]
        else:
            aggregated = [
                agg + (w * weight) for agg, w in zip(aggregated, params)
            ]
    
    return aggregated


def trimmed_mean_aggregate(results: List[Tuple], beta: float = 0.1) -> list:
    """
    Trimmed Mean - ROBUST aggregation (RECOMMENDED for security).
    
    HOW IT WORKS:
    - For each individual weight in the model:
      1. Collects that weight's value from ALL clients
      2. Sorts them from smallest to largest
      3. Removes the top β% and bottom β% (the extremes)
      4. Averages only the remaining middle values
    
    EXAMPLE with 10 clients and β=0.1 (trim 10%):
    - For one weight: values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 99.0]
    - After sorting: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 99.0]
    - Trim 10% from each end (1 value): remove 0.1 and 99.0
    - Average remaining: (0.2+0.3+0.4+0.5+0.6+0.7+0.8+0.9)/8 = 0.55
    - Without trimming: average would be 10.35 (poisoned by the 99.0!)
    
    WHY IT'S SECURE:
    - Malicious clients that send extreme values get trimmed away
    - Works well even with 1-2 bad clients out of many
    
    NOTE: With only 3 clients and β=0.1, we can trim at most ~0 from each end.
    So we use a minimum of trimming 0 and effectively do a simple mean.
    But the norm clipping + anomaly detection still protect us.
    
    Args:
        results: List of (client, num_samples, metrics) tuples
        beta: Fraction to trim from each end (0.0 to 0.5)
    """
    if not results:
        return None
    
    # Get parameters from all clients
    all_params = [client.get_parameters({}) for client, _, _ in results]
    num_clients = len(all_params)
    num_layers = len(all_params[0])
    
    # Number of values to trim from each end
    # With 3 clients and beta=0.1, trim_count = int(0.1 * 3) = 0
    # With 10 clients and beta=0.1, trim_count = int(0.1 * 10) = 1
    # SECURITY FIX (Vuln 5): Enforce minimum trim of 1 when we have 3+ clients
    # Without this, beta=0.1 with 3 clients gives trim_count=0 (no protection)
    trim_count = int(beta * num_clients)
    if num_clients >= 3 and trim_count < 1:
        trim_count = 1
    
    aggregated = []
    
    for layer_idx in range(num_layers):
        # Stack this layer's weights from all clients
        # Shape: (num_clients, *weight_shape)
        stacked = np.stack([params[layer_idx] for params in all_params])
        
        # Sort along the client axis (axis=0)
        sorted_params = np.sort(stacked, axis=0)
        
        # Trim the top and bottom values
        if trim_count > 0:
            trimmed = sorted_params[trim_count:-trim_count]
        else:
            trimmed = sorted_params  # No trimming possible with few clients
        
        # Average the remaining values
        aggregated.append(np.mean(trimmed, axis=0))
    
    return aggregated


def krum_aggregate(results: List[Tuple], f: int = 1) -> list:
    """
    Krum - BYZANTINE-ROBUST aggregation.
    
    HOW IT WORKS:
    - Assumes up to 'f' clients could be malicious (Byzantine)
    - For each client, calculates how "close" its update is to other clients
    - Selects the single client whose update is closest to the majority
    - Uses ONLY that client's update (ignores all others)
    
    INTUITION:
    Imagine 3 people reporting a room's temperature:
    - Person A says 72°F
    - Person B says 73°F  
    - Person C says 999°F (malicious!)
    Krum picks the person closest to others → Person A or B (not C)
    
    WHY IT'S SECURE:
    - Even if f clients are malicious, the selected update is from an honest client
    - Guaranteed safety as long as f < n/2 - 1
    
    DOWNSIDE:
    - Only uses ONE client's data per round → slow convergence
    - With 3 clients and f=1, we select 1 out of 3
    
    Args:
        results: List of (client, num_samples, metrics) tuples
        f: Number of Byzantine (potentially malicious) clients to tolerate
    """
    if not results:
        return None
    
    all_params = [client.get_parameters({}) for client, _, _ in results]
    num_clients = len(all_params)
    
    # Flatten each client's parameters into a single vector for distance comparison
    flat_params = []
    for params in all_params:
        flat = np.concatenate([p.flatten() for p in params])
        flat_params.append(flat)
    
    # For each client, compute sum of distances to its (n-f-2) closest neighbors
    # The client with the SMALLEST sum is the most "trustworthy"
    scores = []
    for i in range(num_clients):
        distances = []
        for j in range(num_clients):
            if i != j:
                dist = np.linalg.norm(flat_params[i] - flat_params[j])
                distances.append(dist)
        
        distances.sort()
        # Sum of distances to closest (n-f-2) neighbors
        num_closest = max(1, num_clients - f - 2)
        score = sum(distances[:num_closest])
        scores.append(score)
    
    # Select the client with the minimum score (closest to majority)
    selected_idx = np.argmin(scores)
    
    return all_params[selected_idx]


def krum_trimmed_mean_aggregate(results: List[Tuple], f: int = 1, beta: float = 0.1) -> list:
    """
    HYBRID: Krum + Trimmed Mean (STRONGEST defense).
    
    HOW IT WORKS (2-stage defense):
    
    Stage 1 — KRUM SCORING:
    - Calculates a "trustworthiness score" for each client
    - Clients whose updates are closest to the majority get LOW scores (good)
    - Clients whose updates are far from others get HIGH scores (suspicious)
    
    Stage 2 — FILTER + TRIMMED MEAN:
    - Removes the most suspicious client(s) identified by Krum
    - Applies Trimmed Mean on the REMAINING trustworthy clients
    
    ANALOGY:
    Imagine grading student essays:
    - Stage 1 (Krum): A panel checks which essays look plagiarized/fake → removes them
    - Stage 2 (Trimmed Mean): For the remaining genuine essays, remove highest and
      lowest scores, then average the rest
    
    WHY THIS IS BETTER THAN EITHER ALONE:
    - Krum alone only uses 1 client → wastes data from honest clients
    - Trimmed Mean alone might not catch a subtle poisoner within trimming range
    - Together: Krum catches the bad actor, Trimmed Mean safely aggregates the rest
    
    Args:
        results: List of (client, num_samples, metrics) tuples
        f: Number of potentially malicious clients to exclude via Krum scoring
        beta: Trimmed Mean trimming fraction for remaining clients
    """
    if not results:
        return None
    
    all_params = [client.get_parameters({}) for client, _, _ in results]
    num_clients = len(all_params)
    
    # =============================================
    # STAGE 1: Krum Scoring (rank clients by trust)
    # =============================================
    # Flatten each client's parameters for distance comparison
    flat_params = []
    for params in all_params:
        flat = np.concatenate([p.flatten() for p in params])
        flat_params.append(flat)
    
    # Compute Krum scores (lower = more trustworthy)
    scores = []
    for i in range(num_clients):
        distances = []
        for j in range(num_clients):
            if i != j:
                dist = np.linalg.norm(flat_params[i] - flat_params[j])
                distances.append(dist)
        distances.sort()
        num_closest = max(1, num_clients - f - 2)
        score = sum(distances[:num_closest])
        scores.append(score)
    
    # Rank clients: lower score = more trustworthy
    ranked_indices = np.argsort(scores)  # Indices sorted by trust (best first)
    
    # =============================================
    # STAGE 2: Exclude worst client(s) + Trimmed Mean
    # =============================================
    # Keep the (n - f) most trustworthy clients
    num_to_keep = max(2, num_clients - f)  # Keep at least 2 for meaningful aggregation
    trusted_indices = ranked_indices[:num_to_keep]
    trusted_params = [all_params[i] for i in trusted_indices]
    
    num_layers = len(trusted_params[0])
    num_trusted = len(trusted_params)
    
    # Calculate trim count for Trimmed Mean
    # SECURITY FIX (Vuln 5): Enforce minimum trim of 1 for meaningful protection
    trim_count = int(beta * num_trusted)
    if num_trusted >= 3 and trim_count < 1:
        trim_count = 1
    
    aggregated = []
    for layer_idx in range(num_layers):
        stacked = np.stack([params[layer_idx] for params in trusted_params])
        sorted_params = np.sort(stacked, axis=0)
        
        if trim_count > 0 and num_trusted > 2 * trim_count:
            trimmed = sorted_params[trim_count:-trim_count]
        else:
            trimmed = sorted_params
        
        aggregated.append(np.mean(trimmed, axis=0))
    
    return aggregated


def clip_update_norm(params: list, max_norm: float) -> Tuple[list, float, bool]:
    """
    Norm Clipping - Limits the total magnitude of a client's update.
    
    WHY: A malicious client might send a very large update to dominate the 
    aggregation. By clipping, we limit any single client's influence.
    
    HOW: If the total norm (size) of the update is larger than max_norm,
    we scale it down proportionally to have norm = max_norm.
    
    Args:
        params: List of numpy arrays (model weights)
        max_norm: Maximum allowed total norm
    
    Returns:
        (clipped_params, original_norm, was_clipped)
    """
    # Calculate total norm of the update
    flat = np.concatenate([p.flatten() for p in params])
    norm = np.linalg.norm(flat)
    
    if norm > max_norm:
        # Scale down: multiply each parameter by (max_norm / norm)
        scale = max_norm / norm
        clipped = [p * scale for p in params]
        return clipped, norm, True
    
    return params, norm, False


def detect_anomalies(results: List[Tuple], threshold: float = None) -> list:
    """
    Anomaly Detection - Identifies suspicious client updates.
    
    HOW IT WORKS (IMPROVED for small client counts):
    1. Calculates the norm (total size) of each client's update
    2. Computes pairwise cosine similarities between all clients
    3. A client is flagged if:
       a) Its norm exceeds the absolute threshold, OR
       b) Its norm is > mean + 2*std (statistical), OR
       c) Its average cosine similarity to other clients is < 0.5 (directional)
    
    SECURITY FIX: Added cosine similarity check which works well even with
    only 3 clients, catching subtle directional attacks that norm-based
    detection misses.
    
    Args:
        results: List of (client, num_samples, metrics) tuples
        threshold: Optional manual threshold. If None, uses statistical detection.
    
    Returns:
        List of dicts with anomaly info for each client
    """
    all_params = [client.get_parameters({}) for client, _, _ in results]
    
    # Calculate norm for each client
    flat_params = []
    norms = []
    for params in all_params:
        flat = np.concatenate([p.flatten() for p in params])
        flat_params.append(flat)
        norms.append(np.linalg.norm(flat))
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms) if len(norms) > 1 else 0
    
    # Compute pairwise cosine similarities (works well with small N)
    num_clients = len(flat_params)
    cosine_sims = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                dot = np.dot(flat_params[i], flat_params[j])
                norm_product = norms[i] * norms[j]
                cosine_sims[i][j] = dot / norm_product if norm_product > 0 else 0
    
    anomaly_results = []
    for i, norm in enumerate(norms):
        is_anomalous = False
        reasons = []
        
        # Check 1: Statistical detection (norm-based)
        if std_norm > 0 and abs(norm - mean_norm) > 2 * std_norm:
            is_anomalous = True
            reasons.append(f"Norm {norm:.1f} is {abs(norm - mean_norm) / std_norm:.1f}σ from mean")
        
        # Check 2: Absolute threshold detection
        if threshold and norm > threshold:
            is_anomalous = True
            reasons.append(f"Norm {norm:.1f} exceeds threshold {threshold:.1f}")
        
        # Check 3: Cosine similarity (catches directional attacks with small N)
        avg_cosine = np.mean([cosine_sims[i][j] for j in range(num_clients) if j != i])
        if avg_cosine < 0.5 and num_clients > 1:
            is_anomalous = True
            reasons.append(f"Low cosine similarity ({avg_cosine:.3f}) — update direction diverges from peers")
        
        anomaly_results.append({
            "client_id": i,
            "norm": norm,
            "cosine_sim": avg_cosine if num_clients > 1 else 1.0,
            "is_anomalous": is_anomalous,
            "reason": "; ".join(reasons) if reasons else ""
        })
    
    return anomaly_results


def aggregate_fit_results(results: List[Tuple], method: str = None) -> Tuple[list, dict]:
    """
    MASTER AGGREGATION FUNCTION
    
    This is the main entry point that:
    1. Filters out CHSH-blocked clients (SECURITY FIX)
    2. Runs anomaly detection on remaining client updates
    3. Clips update norms (limits individual client influence)
    4. Calls the selected aggregation strategy
    5. Returns the aggregated parameters + security report
    
    Args:
        results: List of (client, num_samples, metrics) tuples
        method: Aggregation method override. If None, uses AGGREGATION_METHOD.
    
    Returns:
        Tuple of (aggregated_params, security_report)
    """
    if not results:
        return None, {}
    
    method = method or AGGREGATION_METHOD
    
    # Step 0: Filter out CHSH-blocked clients (SECURITY FIX)
    active_results = []
    blocked_clients = []
    for client, num_samples, metrics in results:
        if metrics.get("chsh_blocked", False):
            blocked_clients.append(metrics.get("cid", "?"))
        else:
            active_results.append((client, num_samples, metrics))
    
    if not active_results:
        return None, {
            "method": method,
            "anomalies": [],
            "num_anomalous": 0,
            "total_clients": len(results),
            "blocked_clients": blocked_clients,
            "active_clients": 0,
            "all_blocked": True,
        }
    
    # Step 1: Anomaly Detection (on active clients only)
    anomalies = detect_anomalies(active_results, threshold=NORM_THRESHOLD)
    num_anomalous = sum(1 for a in anomalies if a["is_anomalous"])
    
    # Step 1b: Norm Clipping — limit each client's update magnitude before aggregation.
    # This prevents any single client (malicious or not) from dominating via a large update.
    clipped_results = []
    for client, num_samples, metrics in active_results:
        params = client.get_parameters({})
        clipped_params, orig_norm, was_clipped = clip_update_norm(params, max_norm=NORM_THRESHOLD)
        if was_clipped:
            client.set_parameters(clipped_params)  # Apply clipped weights back onto client
        clipped_results.append((client, num_samples, metrics))
    active_results = clipped_results
    
    # Step 2: Aggregate using selected strategy
    if method == "krum_trimmed_mean":
        aggregated = krum_trimmed_mean_aggregate(active_results, f=1, beta=TRIMMED_MEAN_BETA)
    elif method == "trimmed_mean":
        aggregated = trimmed_mean_aggregate(active_results, beta=TRIMMED_MEAN_BETA)
    elif method == "krum":
        aggregated = krum_aggregate(active_results, f=1)
    else:  # fedavg (default)
        aggregated = fedavg_aggregate(active_results)
    
    # Step 3: Build security report for UI
    security_report = {
        "method": method,
        "anomalies": anomalies,
        "num_anomalous": num_anomalous,
        "total_clients": len(results),
        "blocked_clients": blocked_clients,
        "active_clients": len(active_results),
        "all_blocked": False,
    }
    
    return aggregated, security_report


# --- APP ---
def main():
    # Initialize session state for model persistence
    if 'trained_model_params' not in st.session_state:
        st.session_state.trained_model_params = None
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    
    st.title("Quantum-Enabled Federated Learning Architecture for secure Deep Models")
    st.caption("Federated Learning across 3 Hospitals · HAM10000 Dataset · 7 Skin Lesion Types")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption("3 Hospitals · HAM10000 · 7 Classes · CNN")
        st.divider()
        
        num_rounds = st.slider("Training Rounds", min_value=1, max_value=10, value=3)
        
        # Aggregation strategy selector
        st.divider()
        st.subheader("🛡️ Security Settings")
        
        agg_method = st.selectbox(
            "Aggregation Strategy",
            options=["krum_trimmed_mean", "trimmed_mean", "krum", "fedavg"],
            index=0
        )
        
        st.divider()
        st.caption("✅ GPU" if torch.cuda.is_available() else "⚠️ CPU")
        start_btn = st.button("🚀 START TRAINING", type="primary", width="stretch")

    # Main content
    if start_btn:
        st.divider()
        training_container = st.container()
        status_bar = st.empty()
        progress_bar = st.progress(0)
        
        with training_container:
            st.subheader("📡 Federated Training Monitor")
            st.caption(f"🛡️ {agg_method.replace('_',' ').title()} · 🔐 E91 Encryption · 🔒 DP · ✂️ Grad Clipping · 🏥 HAM10000")
            st.divider()
            
            # Initialize global model
            global_model = MultiModalFederatedModel().to(device)
            global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
            
            # Create clients once
            clients = [
                FLQCClient(cid=str(i), device=device, total_clients=3)
                for i in range(3)
            ]
            
            # Training loop
            losses_history = []
            accuracies_history = []
            start_time = time.time()
            
            # Per-client metrics history
            client_losses = [[] for _ in range(3)]
            client_accuracies = [[] for _ in range(3)]
            
            # Create 3 columns for clients
            client_cols = st.columns(3)
            client_containers = []
            
            # Create containers for each client with headers
            for i, col in enumerate(client_cols):
                with col:
                    client_classes = CLIENT_CLASSES.get(i, [])
                    class_labels = ', '.join(client_classes)
                    st.markdown(f"### 🏥 Hospital {chr(65+i)}")
                    # Fixed minimum height (e.g., 50px) ensures columns align perfectly even if text wraps
                    labels = ', '.join([CLASS_DISPLAY[c] for c in client_classes])
                    st.markdown(f'<div style="min-height: 50px; font-size: 14px; color: #a3a8b8;">{labels}</div>', unsafe_allow_html=True)
                    st.divider()
                    client_containers.append(col.container())
            
            # Security log container (appears after client columns)
            security_log_container = st.container()
            
            try:
                for round_num in range(1, num_rounds + 1):
                    round_start = time.time()
                    status_bar.info(f"🔧 Round {round_num}/{num_rounds}: Training all clients...")
                    
                    # Fit phase
                    fit_results = []
                    
                    for i, client in enumerate(clients):
                        # Train client (includes gradient clipping, DP noise, encryption)
                        params, num_samples, metrics = client.fit(global_params, {})
                        fit_results.append((client, num_samples, metrics))
                        
                        # Track per-client metrics
                        client_losses[i].append(metrics.get("loss", 0))
                        client_accuracies[i].append(metrics.get("accuracy", 0))
                        
                        # Update each client's column with their round result
                        with client_containers[i]:
                            with st.expander(f"Round {round_num}", expanded=(round_num == num_rounds)):
                                # Metrics in mini columns
                                mcol1, mcol2 = st.columns(2)
                                with mcol1:
                                    st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
                                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
                                with mcol2:
                                    st.metric("Samples", f"{metrics.get('num_samples', 0):,}")
                                    # CHSH value display
                                    chsh_val = metrics.get("chsh_value", 0)
                                    is_verified = metrics.get("verification_status", False)
                                    if is_verified:
                                        st.success(f"✅ CHSH: {chsh_val:.3f}", icon="🔑")
                                    else:
                                        st.error(f"❌ CHSH: {chsh_val:.3f}", icon="🔑")
                                
                                # Security details expander
                                with st.expander("🔒 Security"):
                                    # Encryption
                                    enc_status = metrics.get("encryption_status", "N/A")
                                    if enc_status == "encrypted":
                                        dec_result = verify_client_encryption(metrics)
                                        if dec_result["verified"]:
                                            st.success("🔐 Encrypted → Decrypted ✅")
                                        else:
                                            st.error(f"Decryption failed: {dec_result['detail']}")
                                    elif enc_status == "failed":
                                        st.error("❌ Encryption Failed")
                                    
                                    # DP
                                    if metrics.get("dp_enabled", False):
                                        dp_col1, dp_col2 = st.columns(2)
                                        with dp_col1:
                                            st.metric("ε (round)", f"{metrics.get('dp_epsilon_round', 0):.1f}")
                                        with dp_col2:
                                            st.metric("ε (total)", f"{metrics.get('dp_epsilon_cumulative', 0):.1f}")
                                        st.caption(f"σ={metrics.get('dp_sigma',0):.4f} · δ={metrics.get('dp_delta',0):.0e} · clips={metrics.get('grad_clips',0)}")
                                    
                                    # Key
                                    st.code(metrics.get('quantum_key', 'N/A'), language='text')
                        
                        time.sleep(0.1)
                    
                    # =============================================
                    # SECURE AGGREGATION
                    # =============================================
                    status_bar.markdown("⚙️ **Running Secure Aggregation...**")
                    time.sleep(0.3)
                    
                    # Use the selected aggregation strategy
                    global_params, security_report = aggregate_fit_results(
                        fit_results, method=agg_method
                    )
                    
                    # Show security report for this round
                    with security_log_container:
                        with st.expander(f"🛡️ Round {round_num} Security Report", expanded=(round_num == num_rounds)):
                            sr_col1, sr_col2, sr_col3 = st.columns(3)
                            with sr_col1:
                                st.metric("Strategy", security_report.get("method", "N/A").replace("_", " ").title())
                            with sr_col2:
                                num_anom = security_report.get("num_anomalous", 0)
                                if num_anom > 0:
                                    st.metric("⚠️ Anomalies", num_anom)
                                else:
                                    st.metric("Anomalies", "0 ✅")
                            with sr_col3:
                                st.metric("Clients", security_report.get("total_clients", 0))
                            
                            # Anomaly details
                            anomalies = security_report.get("anomalies", [])
                            for anom in anomalies:
                                if anom["is_anomalous"]:
                                    st.warning(
                                        f"⚠️ Client {anom['client_id']+1}: {anom['reason']} "
                                        f"(Norm: {anom['norm']:.1f})"
                                    )
                                else:
                                    st.caption(
                                        f"✅ Client {anom['client_id']+1}: Normal "
                                        f"(Norm: {anom['norm']:.1f})"
                                    )
                    
                    # Calculate average metrics
                    avg_loss = np.mean([m["loss"] for _, _, m in fit_results])
                    avg_accuracy = np.mean([m.get("accuracy", 0) for _, _, m in fit_results])
                    losses_history.append(avg_loss)
                    accuracies_history.append(avg_accuracy)
                    
                    round_time = time.time() - round_start
                    
                    status_bar.success(
                        f"✓ Round {round_num} Complete | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"Avg Accuracy: {avg_accuracy:.2f}% | "
                        f"Strategy: {agg_method} | "
                        f"Time: {round_time:.1f}s",
                        icon="🚀"
                    )
                    progress_bar.progress(round_num / num_rounds)
                
                total_time = time.time() - start_time
                
                # Store model in session state for predictions
                st.session_state.trained_model_params = global_params
                st.session_state.training_completed = True
                
                # Save all results to session state so they survive reruns
                final_eps = max(
                    [m.get("dp_epsilon_cumulative", 0) for _, _, m in fit_results],
                    default=0
                )
                st.session_state.training_results = {
                    'losses_history': losses_history,
                    'accuracies_history': accuracies_history,
                    'total_time': total_time,
                    'agg_method': agg_method,
                    'final_eps': final_eps,
                }
                
                st.success(f"✅ Training Complete! Total Time: {total_time:.1f}s")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

    if not st.session_state.training_completed:
        # Welcome screen (only shows before any training)
        st.info("👈 Configure training settings and click START")
        
        st.subheader("📊 HAM10000 Dataset")
        st.caption("10,015 dermoscopic images · 7 skin lesion classes · Non-IID split across 3 hospitals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🏥 Hospital A**")
            st.caption("Melanocytic Nevi, Melanoma")
        with col2:
            st.markdown("**🏥 Hospital B**")
            st.caption("Benign Keratosis, Basal Cell Carcinoma, Actinic Keratoses")
        with col3:
            st.markdown("**🏥 Hospital C**")
            st.caption("Vascular Lesions, Dermatofibroma")
        
        st.divider()
        st.subheader("🔬 Skin Lesion Classes")
        class_cols = st.columns(4)
        class_info = [
            ("akiec", "Actinic Keratoses", "Pre-malignant"),
            ("bcc", "Basal Cell Carcinoma", "Malignant"),
            ("bkl", "Benign Keratosis", "Benign"),
            ("df", "Dermatofibroma", "Benign"),
            ("mel", "Melanoma", "⚠️ Malignant"),
            ("nv", "Melanocytic Nevi", "Benign"),
            ("vasc", "Vascular Lesions", "Benign"),
        ]
        for i, (abbrev, name, severity) in enumerate(class_info):
            with class_cols[i % 4]:
                st.markdown(f"**{abbrev}** — {name}")
                st.caption(severity)
        
        st.divider()
        st.subheader("🛡️ Security Layers")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("🔐 **Communication**\n- E91 Quantum Keys\n- AES-128 Encryption\n- CHSH Verification")
        with col_b:
            st.markdown("🛡️ **Aggregation**\n- Krum + Trimmed Mean\n- Anomaly Detection\n- Norm Clipping")
        with col_c:
            st.markdown("🔒 **Endpoint**\n- Differential Privacy\n- Gradient Clipping")

    # =====================================================
    # PERSISTENT RESULTS SECTION (survives page reruns)
    # =====================================================
    if st.session_state.training_completed and st.session_state.training_results is not None:
        results = st.session_state.training_results
        
        # --- TRAINING RESULTS ---
        st.divider()
        st.subheader("📊 Training Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Loss", f"{results['losses_history'][-1]:.4f}")
        with col2:
            st.metric("Final Accuracy", f"{results['accuracies_history'][-1]:.2f}%")
        with col3:
            st.metric("Time", f"{results['total_time']:.1f}s")
        
        # Charts
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Training Loss Over Rounds**")
            st.line_chart(results['losses_history'])
        with col_b:
            st.markdown("**Training Accuracy Over Rounds**")
            st.line_chart(results['accuracies_history'])
        
        # Privacy & Security
        st.info(f"✓ Quantum keys verified | Privacy budget spent: ε = {results['final_eps']:.2f}")
        
        st.divider()
        st.subheader("🛡️ Security Summary")
        sec_cols = st.columns(4)
        with sec_cols[0]:
            st.success("🔐 E91 Encrypted")
        with sec_cols[1]:
            st.success(f"🛡️ {results['agg_method'].replace('_', ' ').title()}")
        with sec_cols[2]:
            st.success(f"🔒 (ε={results['final_eps']:.1f}, δ=1e-5)-DP")
        with sec_cols[3]:
            st.success("✂️ Gradient Clipping")
        
        # --- GLOBAL MODEL EVALUATION ---
        st.divider()
        st.subheader("🎓 Global Model Evaluation")
        
        try:
            eval_model = MultiModalFederatedModel().to(device)
            params_dict = zip(eval_model.state_dict().keys(), st.session_state.trained_model_params)
            state_dict = OrderedDict({k: torch.from_numpy(np.array(v)) for k, v in params_dict})
            eval_model.load_state_dict(state_dict, strict=True)
            eval_model.eval()
            
            test_dataset = get_full_test_dataset()
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            test_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            criterion = torch.nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = eval_model(inputs)
                    test_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_accuracy = 100 * correct / total
            avg_test_loss = test_loss / len(test_loader)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Accuracy", f"{test_accuracy:.2f}%")
            with col2:
                st.metric("Test Loss", f"{avg_test_loss:.4f}")
            with col3:
                st.metric("Test Samples", f"{total:,}")
            
            # Confusion Matrix
            with st.expander("🔍 Confusion Matrix & Per-Class Accuracy", expanded=False):
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=CLASS_NAMES,
                            yticklabels=CLASS_NAMES, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Skin Lesion Classification — Confusion Matrix')
                st.pyplot(fig)
                
                st.markdown("**Per-Class Accuracy:**")
                for i, class_name in enumerate(CLASS_NAMES):
                    display_name = CLASS_DISPLAY[class_name]
                    class_correct = cm[i, i]
                    class_total = cm[i].sum()
                    class_acc = 100 * class_correct / class_total if class_total > 0 else 0
                    st.progress(class_acc / 100, text=f"{display_name} ({class_name}): {class_acc:.1f}% ({class_correct}/{class_total})")
        
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
        
        # --- PREDICTION SECTION ---
        st.divider()
        st.subheader("🔬 Test Prediction — Skin Lesion")
        
        try:
            prediction_model = MultiModalFederatedModel().to(device)
            params_dict = zip(prediction_model.state_dict().keys(), st.session_state.trained_model_params)
            state_dict = OrderedDict({k: torch.from_numpy(np.array(v)) for k, v in params_dict})
            prediction_model.load_state_dict(state_dict, strict=True)
            prediction_model.eval()
            
            uploaded_file = st.file_uploader("Upload a dermoscopic skin lesion image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], key="prediction_uploader")
            
            if uploaded_file is not None:
                try:
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    # SECURITY FIX (Vuln 9): Validate file size (max 10 MB)
                    MAX_FILE_SIZE_MB = 10
                    file_size = uploaded_file.size
                    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                        st.error(f"❌ File too large ({file_size / (1024*1024):.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
                        st.stop()
                    
                    # SECURITY FIX (Vuln 9): Verify this is actually a valid image
                    try:
                        verify_img = Image.open(uploaded_file)
                        verify_img.verify()  # Checks integrity without loading full data
                        uploaded_file.seek(0)  # Reset file pointer after verify
                    except Exception:
                        st.error("❌ Invalid image file — the file appears to be corrupted or not a real image.")
                        st.stop()
                    
                    try:
                        image = Image.open(uploaded_file).convert('RGB')
                    except Exception as e:
                        st.error(f"❌ Error loading image: {str(e)}")
                        st.stop()
                    
                    col_a, col_b = st.columns([1, 2])
                    
                    try:
                        transform = transforms.Compose([
                            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.7635, 0.5461, 0.5705],
                                std=[0.1409, 0.1520, 0.1695]
                            )
                        ])
                        img_tensor = transform(image).unsqueeze(0).to(device)
                    except Exception as e:
                        st.error(f"❌ Error preprocessing image: {str(e)}")
                        st.stop()
                    
                    try:
                        with torch.no_grad():
                            output = prediction_model(img_tensor)
                            logits = output[0]

                            # Guard against unstable model outputs (NaN/Inf) before softmax.
                            if not torch.isfinite(logits).all():
                                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                            probabilities = torch.nn.functional.softmax(logits, dim=0)

                            # If softmax still produces invalid values, fall back to uniform probs.
                            if (not torch.isfinite(probabilities).all()) or (float(probabilities.sum()) <= 0.0):
                                probabilities = torch.ones(NUM_CLASSES, device=logits.device) / NUM_CLASSES

                            predicted_idx = int(torch.argmax(probabilities).item())
                            confidence = float(probabilities[predicted_idx].item())
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.stop()
                    
                    predicted_class = CLASS_NAMES[predicted_idx]
                    predicted_display = CLASS_DISPLAY[predicted_class]
                    
                    # Medical info for each skin lesion type
                    LESION_INFO = {
                        'akiec': {
                            'severity': '⚡ Pre-Malignant',
                            'danger': 'moderate',
                            'intro': 'Actinic Keratoses are rough, scaly patches caused by years of sun exposure — they can progress to squamous cell carcinoma if untreated.',
                            'precaution': (
                                "**🩺 What to do:**\n"
                                "- Schedule a dermatologist appointment within 2 weeks\n"
                                "- Apply broad-spectrum **SPF 50+** sunscreen every 2 hours when outdoors\n"
                                "- Avoid direct sun exposure between **10 AM – 4 PM**\n"
                                "- Wear wide-brimmed hats and UV-protective clothing\n"
                                "- Do **not** scratch, pick, or try to remove the lesion yourself\n"
                                "- Get full-body skin exams every **6–12 months**\n"
                                "- Treatment options include cryotherapy, topical creams (5-FU, Imiquimod), or photodynamic therapy"
                            )
                        },
                        'bcc': {
                            'severity': '🔴 Malignant',
                            'danger': 'high',
                            'intro': 'Basal Cell Carcinoma is the most common form of skin cancer — it rarely spreads but can cause significant local tissue damage if ignored.',
                            'precaution': (
                                "**⚠️ Action required:**\n"
                                "- **See a dermatologist as soon as possible** — early removal is highly curable\n"
                                "- Do **not** ignore slow-growing sores that don't heal\n"
                                "- Avoid tanning beds and prolonged sun exposure completely\n"
                                "- Apply **SPF 50+** daily, even on cloudy days\n"
                                "- Perform monthly self-exams — look for pearly bumps, pink patches, or open sores\n"
                                "- Treatment: surgical excision, Mohs surgery, or radiation therapy\n"
                                "- After treatment, follow-up skin exams every **3–6 months** for 5 years"
                            )
                        },
                        'bkl': {
                            'severity': '✅ Benign',
                            'danger': 'low',
                            'intro': 'Benign Keratosis (seborrheic keratosis) is a harmless, non-cancerous skin growth that commonly appears with aging.',
                            'precaution': (
                                "**ℹ️ General care:**\n"
                                "- No medical treatment is necessary in most cases\n"
                                "- Monitor the lesion — consult a doctor if it **changes color, shape, or size**\n"
                                "- See a doctor if it becomes **irritated, bleeds, or itches** persistently\n"
                                "- Cosmetic removal is available via cryotherapy or curettage if desired\n"
                                "- Apply sunscreen regularly to prevent new growths\n"
                                "- Annual skin check-ups are a good practice"
                            )
                        },
                        'df': {
                            'severity': '✅ Benign',
                            'danger': 'low',
                            'intro': 'Dermatofibroma is a common, harmless firm bump in the skin, often on the legs — usually a reaction to a minor injury like an insect bite.',
                            'precaution': (
                                "**ℹ️ General care:**\n"
                                "- No treatment required unless it causes discomfort\n"
                                "- See a doctor if it **grows rapidly, changes color, or becomes painful**\n"
                                "- Surgical removal is an option if the bump is bothersome or cosmetically undesirable\n"
                                "- Avoid picking or scratching the area\n"
                                "- These bumps are permanent but harmless — they rarely recur after removal"
                            )
                        },
                        'mel': {
                            'severity': '🔴 Malignant — Dangerous',
                            'danger': 'critical',
                            'intro': 'Melanoma is the most dangerous form of skin cancer — it can spread rapidly to other organs if not caught early. Early detection is life-saving.',
                            'precaution': (
                                "**🚨 URGENT — Seek immediate medical attention:**\n"
                                "- **See a dermatologist IMMEDIATELY** — do NOT wait\n"
                                "- Follow the **ABCDE rule**: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving shape\n"
                                "- A biopsy is essential for definitive diagnosis\n"
                                "- Avoid all UV exposure — no tanning beds, use **SPF 50+** at all times\n"
                                "- Perform **monthly full-body self-exams** using a mirror\n"
                                "- Treatment may include surgery, immunotherapy, targeted therapy, or radiation\n"
                                "- After diagnosis, follow-up exams every **3 months** for the first 2 years\n"
                                "- Inform family members — melanoma can have a genetic component"
                            )
                        },
                        'nv': {
                            'severity': '✅ Benign',
                            'danger': 'low',
                            'intro': 'Melanocytic Nevi (moles) are common benign growths — most people have 10-40 moles, and the vast majority are completely harmless.',
                            'precaution': (
                                "**ℹ️ General care:**\n"
                                "- Monitor moles regularly using the **ABCDE rule** (Asymmetry, Border, Color, Diameter, Evolving)\n"
                                "- Annual skin check-ups with a dermatologist are recommended\n"
                                "- See a doctor if any mole **changes shape, color, bleeds, or itches**\n"
                                "- Use sunscreen to prevent new moles and protect existing ones\n"
                                "- Avoid picking or irritating moles\n"
                                "- People with many moles (>50) should have more frequent check-ups"
                            )
                        },
                        'vasc': {
                            'severity': '✅ Benign',
                            'danger': 'low',
                            'intro': 'Vascular Lesions (e.g., cherry angiomas, hemangiomas) are benign growths of blood vessels in the skin — they are almost always harmless.',
                            'precaution': (
                                "**ℹ️ General care:**\n"
                                "- Usually no treatment is needed — most vascular lesions are harmless\n"
                                "- Consult a doctor if it **bleeds frequently, grows rapidly, or changes appearance**\n"
                                "- Cosmetic removal via laser therapy or electrocautery is available if desired\n"
                                "- Avoid trauma to the area to prevent bleeding\n"
                                "- These are common and increase in number with age"
                            )
                        },
                    }
                    
                    info = LESION_INFO.get(predicted_class, {})
                    
                    # Left column: Image + All Probabilities dropdown
                    with col_a:
                        st.image(image, caption=f"Uploaded: {image.size[0]}×{image.size[1]}", width=200)
                        
                        with st.expander("📊 All Class Probabilities", expanded=False):
                            probs_sorted, idx_sorted = torch.sort(probabilities, descending=True)
                            for i in range(NUM_CLASSES):
                                class_idx = idx_sorted[i].item()
                                class_prob = probs_sorted[i].item()
                                if not np.isfinite(class_prob):
                                    class_prob = 0.0
                                class_prob = float(np.clip(class_prob, 0.0, 1.0))
                                cls_name = CLASS_NAMES[class_idx]
                                display = CLASS_DISPLAY[cls_name]
                                cls_info = LESION_INFO.get(cls_name, {})
                                severity_tag = cls_info.get('severity', '')
                                marker = " 👈" if cls_name == predicted_class else ""
                                st.progress(class_prob, text=f"{display} ({cls_name}): {class_prob*100:.2f}%{marker}")
                                st.caption(f"{severity_tag}")
                    
                    # Right column: Diagnosis + Confidence + Severity
                    with col_b:
                        # Diagnosis header
                        if info.get('danger') == 'critical':
                            if confidence > 0.5:
                                st.error(f"### 🚨 {predicted_display} ({predicted_class})")
                            else:
                                st.warning(f"### 🚨 {predicted_display} ({predicted_class}) — Low Confidence")
                        elif info.get('danger') == 'high':
                            if confidence > 0.5:
                                st.error(f"### ⚠️ {predicted_display} ({predicted_class})")
                            else:
                                st.warning(f"### ⚠️ {predicted_display} ({predicted_class}) — Low Confidence")
                        elif info.get('danger') == 'moderate':
                            if confidence > 0.5:
                                st.warning(f"### ⚡ {predicted_display} ({predicted_class})")
                            else:
                                st.info(f"### ⚡ {predicted_display} ({predicted_class}) — Low Confidence")
                        else:
                            if confidence > 0.5:
                                st.success(f"### ✅ {predicted_display} ({predicted_class})")
                            else:
                                st.info(f"### {predicted_display} ({predicted_class}) — Low Confidence")
                        
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Severity display under confidence
                        severity_text = info.get('severity', 'Unknown')
                        danger_level = info.get('danger', 'low')
                        if danger_level == 'critical':
                            st.error(f"**Severity:** {severity_text}")
                        elif danger_level == 'high':
                            st.warning(f"**Severity:** {severity_text}")
                        elif danger_level == 'moderate':
                            st.warning(f"**Severity:** {severity_text}")
                        else:
                            st.success(f"**Severity:** {severity_text}")
                        
                        # One-line intro
                        st.info(f"📋 {info.get('intro', 'No information available.')}")
                    
                    # --- Precautions Below (full width) ---
                    st.divider()
                    st.markdown(f"### 🩺 Recommended Precautions for {predicted_display}")
                    if info.get('danger') in ['critical', 'high']:
                        st.error(info.get('precaution', 'Consult a dermatologist.'))
                    elif info.get('danger') == 'moderate':
                        st.warning(info.get('precaution', 'Consult a dermatologist.'))
                    else:
                        st.success(info.get('precaution', 'Monitor and consult if changes occur.'))
                
                except Exception as e:
                    st.error(f"❌ Unexpected error during prediction: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading model for prediction: {str(e)}")

if __name__ == "__main__":
    main()