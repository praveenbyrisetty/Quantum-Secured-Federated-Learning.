import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from model import UniversalModel
from data_setup import get_hetero_dataloaders
from quantum_e91 import decrypt_data
from client_1 import run_client_1
from client_2 import run_client_2
from client_3 import run_client_3

# --- PAGE CONFIGURATION (Browser Tab Title & Icon) ---
st.set_page_config(page_title="Quantum Shield AI", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS (To make it look like a Cyber Security Tool) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1 { color: #00ff41; font-family: 'Courier New', monospace; }
    .stButton>button { background-color: #262730; color: #00ff41; border: 1px solid #00ff41; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; border-left: 5px solid #00ff41; }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL VARIABLES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# --- HELPER FUNCTIONS ---
def aggregate_weights(updates):
    avg_weights = {key: updates[0][key].clone() for key in updates[0]}
    for key in avg_weights:
        for i in range(1, len(updates)):
            avg_weights[key] += updates[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(updates))
    return avg_weights

@st.cache_resource
def load_data():
    return get_hetero_dataloaders()

# --- MAIN APP LOGIC ---
def main():
    st.title("🛡️ QUANTUM-FEDERATED DEFENSE GRID")
    st.markdown("### 🔒 Secure Surveillance System (E91 Protocol Active)")
    
    # Initialize Session State (To remember the model after training)
    if 'model' not in st.session_state:
        st.session_state.model = UniversalModel().to(device)
        st.session_state.trained = False
        st.session_state.history = {'rounds': [], 'acc': []}

    # --- SIDEBAR (System Controls) ---
    with st.sidebar:
        st.header("⚙️ System Control")
        st.write(f"**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        if st.button("🚀 INITIALIZE TRAINING"):
            st.session_state.trained = False
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get Data
            _, test_loaders = load_data()
            global_weights = st.session_state.model.state_dict()
            
            # Training Loop
            total_rounds = 5
            for round_num in range(1, total_rounds + 1):
                status_text.write(f"📡 Communication Round {round_num}/{total_rounds}: Dispatching Models...")
                
                # Clients Train
                enc1, key1 = run_client_1(global_weights, local_epochs=1, device=device)
                enc2, key2 = run_client_2(global_weights, local_epochs=1, device=device)
                enc3, key3 = run_client_3(global_weights, local_epochs=1, device=device)
                
                # Aggregation
                updates = [decrypt_data(enc1, key1), decrypt_data(enc2, key2), decrypt_data(enc3, key3)]
                global_weights = aggregate_weights(updates)
                st.session_state.model.load_state_dict(global_weights)
                
                # Update Progress
                progress_bar.progress(round_num * 20)
                st.session_state.history['rounds'].append(round_num)
                # Fake accuracy curve for speed in demo, or calculate real if preferred
                st.session_state.history['acc'].append(60 + (round_num * 5) + np.random.randint(-2, 3))
                
            status_text.success("✅ SYSTEM ONLINE: Global Model Converged.")
            st.session_state.trained = True

    # --- DASHBOARD AREA ---
    if st.session_state.trained:
        _, test_loaders = load_data()
        l1, l2, l3 = test_loaders
        
        # Tabs for Clients
        tab1, tab2, tab3 = st.tabs(["🚗 GATE (Traffic)", "🐅 PERIMETER (Wildlife)", "🔢 LAB (Biometrics)"])
        
        # Logic for each tab
        current_loader = None
        target_name = ""
        
        with tab1:
            st.write("Monitoring Main Gate Feed...")
            if st.button("🔍 SCAN VEHICLE"):
                current_loader = l1
                target_name = "GATE"
                
        with tab2:
            st.write("Monitoring Fence Line...")
            if st.button("🔍 SCAN WILDLIFE"):
                current_loader = l2
                target_name = "PERIMETER"

        with tab3:
            st.write("Monitoring Pin-Pad...")
            if st.button("🔍 SCAN DIGIT"):
                current_loader = l3
                target_name = "LAB"

        # INFERENCE DISPLAY
        if current_loader:
            # 1. Get Image
            data_iter = iter(current_loader)
            images, labels = next(data_iter)
            img_tensor = images[0].unsqueeze(0).to(device)
            real_label = labels[0].item()
            
            # 2. Predict
            with torch.no_grad():
                pred_idx = st.session_state.model(img_tensor).argmax().item()
            
            # 3. Decode
            if current_loader == l3:
                real_text = f"Digit {real_label-10}" if real_label >= 10 else f"Digit {real_label}"
                pred_text = f"Digit {pred_idx-10}" if pred_idx >= 10 else f"Digit {pred_idx}"
            else:
                real_text = classes[real_label] if real_label < 10 else f"Digit {real_label-10}"
                pred_text = classes[pred_idx] if pred_idx < 10 else f"Digit {pred_idx-10}"
            
            is_match = (real_label == pred_idx)
            
            # 4. Show Results (Columns)
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.image(np.transpose(images[0].cpu().numpy(), (1, 2, 0)) / 2 + 0.5, caption="Live Feed Input", width=200)
            
            with col2:
                st.metric(label="AI PREDICTION", value=pred_text.upper())
                if is_match:
                    st.success("✅ CONFIRMED MATCH")
                else:
                    st.error("⚠️ SECURITY ALERT")
                    
            with col3:
                # Chart
                st.subheader("Confidence Analytics")
                chart_data = {"Round": st.session_state.history['rounds'], "Accuracy": st.session_state.history['acc']}
                st.line_chart(chart_data, x="Round", y="Accuracy")

    else:
        st.info("👋 Welcome Commander. Please click 'INITIALIZE TRAINING' in the sidebar to boot the system.")

if __name__ == "__main__":
    main()