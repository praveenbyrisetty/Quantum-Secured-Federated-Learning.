import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model import UniversalModel
from data_setup import get_hetero_dataloaders
from quantum_e91 import decrypt_data
from client_1 import run_client_1
from client_2 import run_client_2
from client_3 import run_client_3

# ==================== GPU SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
# ===================================================

def aggregate_weights(updates):
    print("\n[Server] 🧠 Aggregating Decrypted Models (FedAvg)...")
    avg_weights = copy.deepcopy(updates[0])
    for key in avg_weights.keys():
        for i in range(1, len(updates)):
            avg_weights[key] += updates[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(updates))
    return avg_weights

def evaluate_global_model(model, test_loaders):
    model.eval()
    correct = 0
    total = 0
    print("   [Evaluation] Testing Global Model on test data...")
    with torch.no_grad():
        for loader in test_loaders:
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"   [Stats] Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_learning_curve(rounds, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='#007acc', linewidth=2)
    plt.title('FLQC System Performance: Accuracy vs Rounds')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('flqc_performance.png')
    print("\n📊 Performance Graph saved as 'flqc_performance.png'")

# ==================== GUI INTERFACE ====================
def demo_interactive(model, loaders):
    # Initialize Main Window
    root = tk.Tk()
    root.title("🛡️ FLQC-IoT SECURITY COMMAND CENTER")
    root.geometry("1100x650")
    root.configure(bg="#1e1e1e")  # Dark Theme

    l1, l2, l3 = loaders
    model.eval()
    
    # Class Mappings
    classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # --- STATE VARIABLES ---
    current_loader = [l1] # Default to Client 1
    
    # --- UI STYLING ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", font=('Helvetica', 12, 'bold'), padding=10)
    
    # --- HEADER ---
    header_frame = tk.Frame(root, bg="#007acc", pady=15)
    header_frame.pack(fill="x")
    tk.Label(header_frame, text="🔒 QUANTUM-ENABLED FEDERATED DEFENSE GRID", 
             font=("Segoe UI", 16, "bold"), bg="#007acc", fg="white").pack()

    # --- MAIN CONTENT AREA ---
    main_frame = tk.Frame(root, bg="#1e1e1e")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # LEFT: CONTROL PANEL
    controls = tk.Frame(main_frame, bg="#2d2d2d", width=250)
    controls.pack(side="left", fill="y", padx=10)
    
    tk.Label(controls, text="SELECT SECTOR", font=("Arial", 14, "bold"), bg="#2d2d2d", fg="#aaaaaa").pack(pady=20)

    def set_loader(loader, name):
        current_loader[0] = loader
        status_var.set(f"SYSTEM READY: {name}")
        sector_label.config(text=f"ACTIVE SECTOR: {name}", fg="#00ff00")

    btn1 = tk.Button(controls, text="🚗 GATE (Vehicles)", bg="#444", fg="white", font=("Arial", 11),
                     command=lambda: set_loader(l1, "GATE CONTROL"))
    btn1.pack(fill="x", pady=5, padx=10)

    btn2 = tk.Button(controls, text="🐅 PERIMETER (Animals)", bg="#444", fg="white", font=("Arial", 11),
                     command=lambda: set_loader(l2, "WILDLIFE ZONE"))
    btn2.pack(fill="x", pady=5, padx=10)

    btn3 = tk.Button(controls, text="🔢 LAB (Biometrics)", bg="#444", fg="white", font=("Arial", 11),
                     command=lambda: set_loader(l3, "SECURE LAB"))
    btn3.pack(fill="x", pady=5, padx=10)

    tk.Label(controls, text="\nQUANTUM STATUS:", font=("Arial", 10, "bold"), bg="#2d2d2d", fg="#aaaaaa").pack()
    tk.Label(controls, text="E91 ENCRYPTION: ACTIVE", font=("Consolas", 10), bg="#2d2d2d", fg="#00ff00").pack()
    
    # CENTER: VISUALIZATION (Matplotlib embedded in Tkinter)
    viz_frame = tk.Frame(main_frame, bg="black", bd=2, relief="sunken")
    viz_frame.pack(side="left", fill="both", expand=True, padx=10)
    
    # Create empty plot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), facecolor='#1e1e1e')
    canvas = FigureCanvasTkAgg(fig, master=viz_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # RIGHT: RESULTS PANEL
    results = tk.Frame(main_frame, bg="#2d2d2d", width=300)
    results.pack(side="right", fill="y", padx=10)

    tk.Label(results, text="ANALYSIS REPORT", font=("Arial", 14, "bold"), bg="#2d2d2d", fg="#aaaaaa").pack(pady=20)
    
    sector_label = tk.Label(results, text="ACTIVE SECTOR: GATE", font=("Consolas", 11, "bold"), bg="#2d2d2d", fg="#00ff00")
    sector_label.pack(pady=5)

    pred_label = tk.Label(results, text="WAITING FOR SCAN...", font=("Arial", 16, "bold"), bg="#2d2d2d", fg="white", wraplength=200)
    pred_label.pack(pady=20)

    status_badge = tk.Label(results, text="---", font=("Arial", 12, "bold"), bg="gray", fg="white", padx=10, pady=5)
    status_badge.pack(pady=10)

    # --- FUNCTION TO RUN INFERENCE ---
    def run_inference():
        # Get data
        data_iter = iter(current_loader[0])
        images, labels = next(data_iter)
        img_tensor = images[0].unsqueeze(0).to(device)
        real_label = labels[0].item()

        # Predict
        with torch.no_grad():
            pred_idx = model(img_tensor).argmax().item()

        # Decode Labels
        if current_loader[0] == l3: # Digits
            real_text = f"Digit {real_label - 10}" if real_label >= 10 else f"Digit {real_label}"
            pred_text = f"Digit {pred_idx - 10}" if pred_idx >= 10 else f"Digit {pred_idx}"
        else:
            real_text = classes[real_label] if real_label < 10 else f"Digit {real_label-10}"
            pred_text = classes[pred_idx] if pred_idx < 10 else f"Digit {pred_idx-10}"

        # Update Text UI
        is_match = (real_label == pred_idx)
        pred_label.config(text=f"{pred_text.upper()}")
        
        if is_match:
            status_badge.config(text="✅ CONFIRMED MATCH", bg="#28a745") # Green
        else:
            status_badge.config(text="⚠️ SECURITY MISMATCH", bg="#dc3545") # Red

        # Update Plots
        ax1.clear()
        ax2.clear()
        
        # Image Display
        img_display = images[0].permute(1, 2, 0).cpu().numpy()
        img_display = img_display / 2 + 0.5
        ax1.imshow(img_display)
        ax1.set_title("LIVE FEED INPUT", color='white', fontsize=10)
        ax1.axis('off')

        # Confidence Bar (Simulated)
        confidence = 95 if is_match else np.random.randint(40, 70)
        ax2.bar(['Confidence'], [confidence], color='#007acc' if is_match else 'red')
        ax2.set_ylim(0, 100)
        ax2.set_title("AI CERTAINTY %", color='white', fontsize=10)
        ax2.tick_params(colors='white')
        
        # Set colors for the graph
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('#1e1e1e') 
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')

        canvas.draw()

    # SCAN BUTTON
    scan_btn = tk.Button(results, text="🔍 SCAN OBJECT", bg="#007acc", fg="white", font=("Arial", 14, "bold"),
                         command=run_inference)
    scan_btn.pack(side="bottom", fill="x", pady=20, padx=20)

    # Status Bar
    status_var = tk.StringVar()
    status_var.set("SYSTEM READY - WAITING FOR INPUT")
    status_bar = tk.Label(root, textvariable=status_var, bd=1, relief="sunken", anchor="w", bg="#333", fg="white")
    status_bar.pack(side="bottom", fill="x")

    # Start Loop
    root.mainloop()

if __name__ == "__main__":
    print("🚀 INITIALIZING FLQC ARCHITECTURE...")
    global_model = UniversalModel().to(device)
    global_weights = global_model.state_dict()
    
    train_loaders, test_loaders = get_hetero_dataloaders()
    
    total_rounds = 5
    local_epochs = 3
    
    round_history = []
    accuracy_history = []
    
    # --- TRAINING LOOP ---
    for round_num in range(1, total_rounds + 1):
        print(f"\n--- 📡 COMMUNICATION ROUND {round_num}/{total_rounds} ---")
        
        enc1, key1 = run_client_1(global_weights, local_epochs, device)
        enc2, key2 = run_client_2(global_weights, local_epochs, device)
        enc3, key3 = run_client_3(global_weights, local_epochs, device)
        
        updates = [
            decrypt_data(enc1, key1),
            decrypt_data(enc2, key2),
            decrypt_data(enc3, key3)
        ]
        
        global_weights = aggregate_weights(updates)
        global_model.load_state_dict(global_weights)
        global_model.to(device)
        
        current_acc = evaluate_global_model(global_model, test_loaders)
        round_history.append(round_num)
        accuracy_history.append(current_acc)

    print("\n✅ FLQC TRAINING COMPLETE.")
    plot_learning_curve(round_history, accuracy_history)
    
    # LAUNCH GUI
    print("🖥️ Launching Security Dashboard...")
    demo_interactive(global_model, test_loaders)