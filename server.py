import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from multi_modal_model import MultiModalFederatedModel
from quantum_e91 import decrypt_data
from data_setup import get_image_dataset, TextDataset, TabularDataset
from config_loader import load_config
from client_1 import run_client_1
from client_2 import run_client_2
from client_3 import run_client_3

# --- HELPERS ---
def secure_aggregate(updates):
    # Simple FedAvg
    if not updates: return None
    avg = {k: updates[0][k].clone() for k in updates[0]}
    for k in avg:
        for i in range(1, len(updates)):
            avg[k] += updates[i][k]
        avg[k] = torch.div(avg[k], len(updates))
    return avg

def save_uploaded_file(uploaded_file, filename):
    if not os.path.exists("./data/uploads"): os.makedirs("./data/uploads")
    path = os.path.join("./data/uploads", filename)
    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
    return path

# --- MAIN CONFIG ---
st.set_page_config(page_title="FLQC Universal", layout="wide", page_icon="🌐")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- APP ---
def main():
    st.title("🛡️ FLQC: Secure Multi-Modal System")
    st.markdown("### Image • Text • Sensors")

    # 1. SETUP GLOBAL MODELS
    if 'models' not in st.session_state:
        st.session_state.models = {
            'image': MultiModalFederatedModel('image').to(device),
            'text': MultiModalFederatedModel('text').to(device),
            'tabular': MultiModalFederatedModel('tabular').to(device)
        }
        st.session_state.trained = False

    # 2. SIDEBAR CONFIG
    with st.sidebar:
        st.header("⚙️ Data Configuration")
        
        # C1: Images
        st.subheader("Client 1: Traffic (Images)")
        img_path = st.text_input("Image Folder Path", value="./data/images")
        
        # C2: Text
        st.subheader("Client 2: Security (Logs)")
        txt_file = st.file_uploader("Upload Log File (.txt)", type=['txt'])
        txt_path = "./data/test.txt"
        if txt_file: txt_path = save_uploaded_file(txt_file, "logs.txt")
        
        # C3: Sensors
        st.subheader("Client 3: IoT (Sensors)")
        csv_file = st.file_uploader("Upload Sensor Data (.csv)", type=['csv'])
        csv_path = "./data/table.csv"
        if csv_file: csv_path = save_uploaded_file(csv_file, "sensors.csv")

        if st.button("🚀 START TRAINING"):
            st.session_state.trained = False
            progress = st.progress(0)
            status = st.empty()
            
            rounds = 5
            for r in range(1, rounds+1):
                status.write(f"🔄 Round {r}/{rounds}: Dispatching specialized models...")
                
                # --- TRAFFIC (Image) ---
                w_img = st.session_state.models['image'].state_dict()
                enc1, k1 = run_client_1(w_img, r, device, data_path=img_path)
                dec1 = decrypt_data(enc1, k1)
                new_img = secure_aggregate([dec1]) # Aggregating 1 client (Simulated)
                if new_img: st.session_state.models['image'].load_state_dict(new_img)
                
                # --- SECURITY (Text) ---
                w_txt = st.session_state.models['text'].state_dict()
                enc2, k2 = run_client_2(w_txt, r, device, data_path=txt_path)
                dec2 = decrypt_data(enc2, k2)
                new_txt = secure_aggregate([dec2])
                if new_txt: st.session_state.models['text'].load_state_dict(new_txt)

                # --- IOT (Sensors) ---
                w_tab = st.session_state.models['tabular'].state_dict()
                enc3, k3 = run_client_3(w_tab, r, device, data_path=csv_path)
                dec3 = decrypt_data(enc3, k3)
                new_tab = secure_aggregate([dec3])
                if new_tab: st.session_state.models['tabular'].load_state_dict(new_tab)
                
                progress.progress(int(r/rounds * 100))
            
            status.success("✅ Multi-Modal Training Complete")
            st.session_state.trained = True

    # 3. DASHBOARD
    if st.session_state.trained:
        st.divider()
        t1, t2, t3 = st.tabs(["🚗 Traffic Vision", "📝 Security Logs", "📊 IoT Sensors"])
        
        # TRAFFIC TAB
        with t1:
            if st.button("Scan Camera (Test)"):
                try:
                    ds = get_image_dataset(img_path)
                    if len(ds)>0:
                        img, _ = ds[np.random.randint(0, len(ds))]
                        with torch.no_grad():
                            pred = st.session_state.models['image'](img.unsqueeze(0).to(device)).argmax()
                        st.image(img.permute(1,2,0).numpy()/2+0.5, width=150)
                        st.metric("Detected", ["Plane","Car","Ship","Truck"][pred])
                except Exception as e: st.error(f"Image Load Error: {e}")

        # LOGS TAB
        with t2:
            if st.button("Analyze Log (Test)"):
                ds = TextDataset(path=txt_path)
                if len(ds)>0:
                    txt, lbl = ds[np.random.randint(0, len(ds))]
                    with torch.no_grad():
                        pred = st.session_state.models['text'](txt.unsqueeze(0).to(device)).argmax()
                    st.code(f"Log ID: {txt[0:5]}...")
                    st.metric("Status", "🚨 THREAT" if pred==1 else "✅ SAFE")

        # SENSORS TAB
        with t3:
            if st.button("Read Sensor (Test)"):
                ds = TabularDataset(path=csv_path)
                if len(ds)>0:
                    dat, lbl = ds[np.random.randint(0, len(ds))]
                    with torch.no_grad():
                        pred = st.session_state.models['tabular'](dat.unsqueeze(0).to(device)).argmax()
                    st.bar_chart(dat.numpy())
                    st.metric("System State", ["Normal","Warning","Critical"][pred] if pred<3 else "Unknown")
    else:
        st.info("👈 Configure data sources and start training.")

if __name__ == "__main__":
    main()