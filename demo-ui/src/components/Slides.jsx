import React, { useState, useEffect } from 'react';
import { Shield, Zap, Server, CheckCircle, AlertTriangle, Lock, Unlock, EyeOff } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

// Injecting Database mock since we removed it from App imports
const Database = ({ size, color }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
    <path d="M3 5V19A9 3 0 0 0 21 19V5"></path>
    <path d="M3 12A9 3 0 0 0 21 12"></path>
  </svg>
);


// ==========================================
// Slide 1: IID Data Distribution
// ==========================================
export const DataDistributionSlide = () => {
  const [trainingState, setTrainingState] = useState(0); // 0=idle, 1=training local, 2=sending

  const executeTrainingCycle = () => {
    setTrainingState(1);
    setTimeout(() => setTrainingState(2), 2000); // After 2s local training, send to server
  };

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Layer 0: Core Hybrid Architecture</h2>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem', maxWidth: '900px' }}>
        In a true Hybrid Federated Learning environment, patient data <strong>never</strong> leaves the hospital. Instead of pooling datasets, each of the 3 hospital supercomputers trains its own mini-AI locally on its proprietary dataset, and only transmits mathematically encrypted "upgrades" up to the Central Server.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
        <button 
          onClick={executeTrainingCycle}
          disabled={trainingState > 0}
          style={{
            padding: '1rem 2rem', background: trainingState === 2 ? 'var(--bg-card)' : 'var(--accent-cyan)', 
            color: trainingState === 2 ? 'var(--text-muted)' : '#000', 
            border: 'none', borderRadius: '8px', fontSize: '1.1rem', fontWeight: 'bold',
            cursor: trainingState > 0 ? 'not-allowed' : 'pointer', 
            boxShadow: trainingState === 0 ? '0 0 20px rgba(0,240,255,0.4)' : 'none', 
            transition: 'all 0.3s'
          }}
        >
          {trainingState === 0 ? 'Initialize Local Hospital AI Training' : 
           trainingState === 1 ? 'Hospitals Crunching Data...' : 
           'Encrypted Weights Transmitted Successfully'}
        </button>

        <div style={{ width: '100%', maxWidth: '900px', display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '1rem', position: 'relative' }}>
          
          {/* Top: Central Server */}
          <div className="glass-panel" style={{ 
            padding: '1.5rem 3rem', textAlign: 'center', zIndex: 10,
            border: trainingState === 2 ? '1px solid var(--accent-cyan)' : '1px solid var(--border-glass)',
            boxShadow: trainingState === 2 ? '0 0 30px rgba(0, 240, 255, 0.2)' : 'none',
            transition: 'all 0.5s'
          }}>
            <Server size={48} color={trainingState === 2 ? 'var(--accent-cyan)' : 'var(--accent-violet)'} />
            <h3 style={{ margin: '0.5rem 0' }}>Central Aggregation Server</h3>
            <div style={{ color: trainingState === 2 ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
              {trainingState === 2 ? 'Models Received - Awaiting Aggregation' : 'Awaiting Hospital Outputs...'}
            </div>
          </div>

          {/* Connection Lines (E91 Protocol Animation) */}
          <div style={{ display: 'flex', width: '70%', height: '80px', position: 'relative', marginTop: '-10px', marginBottom: '-10px', zIndex: 1 }}>
             
             {/* E91 Status Badge */}
             <div style={{
                 position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                 background: trainingState === 2 ? 'rgba(0, 0, 0, 0.8)' : 'transparent', 
                 padding: trainingState === 2 ? '0.4rem 1rem' : '0', 
                 borderRadius: '20px', 
                 border: trainingState === 2 ? '1px solid var(--accent-cyan)' : 'none', 
                 color: 'var(--accent-cyan)',
                 fontSize: '0.85rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px', zIndex: 20,
                 boxShadow: trainingState === 2 ? '0 0 15px rgba(0, 240, 255, 0.5)' : 'none',
                 opacity: trainingState === 2 ? 1 : 0, transition: 'all 0.5s ease-in-out'
             }}>
                 <Lock size={14} /> E91 Quantum Encryption Active
             </div>

             {/* Left line */}
             <div style={{ flex: 1, borderTop: '2px dashed var(--border-glass)', borderLeft: '2px dashed var(--border-glass)', borderTopLeftRadius: '16px', marginTop: '40px', position: 'relative', overflow: 'hidden' }}>
                {trainingState === 2 && <div style={{position: 'absolute', bottom: 0, left: '-2px', width: '4px', height: '100%', background: 'linear-gradient(to top, var(--accent-cyan), transparent)', animation: 'slideUpFade 1s infinite'}} />}
             </div>
             {/* Center line */}
             <div style={{ width: '2px', background: 'var(--border-glass)', position: 'relative', overflow: 'hidden' }}>
                {trainingState === 2 && <div style={{position: 'absolute', bottom: 0, left: 0, width: '100%', height: '100%', background: 'linear-gradient(to top, var(--accent-emerald), transparent)', animation: 'slideUpFade 1s infinite 0.2s'}} />}
             </div>
             {/* Right line */}
             <div style={{ flex: 1, borderTop: '2px dashed var(--border-glass)', borderRight: '2px dashed var(--border-glass)', borderTopRightRadius: '16px', marginTop: '40px', position: 'relative', overflow: 'hidden' }}>
                {trainingState === 2 && <div style={{position: 'absolute', bottom: 0, right: '-2px', width: '4px', height: '100%', background: 'linear-gradient(to top, var(--accent-violet), transparent)', animation: 'slideUpFade 1s infinite 0.4s'}} />}
             </div>
          </div>

          {/* Bottom: 3 Hospitals */}
          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginTop: '1rem', zIndex: 10 }}>
            {[1, 2, 3].map(id => (
              <div key={id} className="glass-panel" style={{ 
                padding: '1.5rem', width: '260px', textAlign: 'center',
                borderTop: `4px solid ${id===1?'var(--accent-cyan)':id===2?'var(--accent-emerald)':'var(--accent-violet)'}`
              }}>
                <h3 style={{ marginBottom: '1rem' }}>Hospital Client {['A', 'B', 'C'][id-1]}</h3>
                
                {/* Local Dataset representation */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginBottom: '1rem', padding: '0.8rem', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                  <Database size={24} color="var(--text-muted)" />
                  <div style={{ textAlign: 'left', fontSize: '0.9rem' }}>
                    <div style={{fontWeight: 'bold'}}>Local HAM10000 subset</div>
                    <div style={{color: 'var(--text-muted)'}}>Locked & Secured</div>
                  </div>
                </div>

                {/* Training status */}
                <div style={{ 
                  padding: '0.5rem', borderRadius: '4px', fontSize: '0.9rem', fontWeight: 'bold',
                  background: trainingState === 0 ? 'rgba(255,255,255,0.05)' : trainingState === 1 ? 'rgba(0,240,255,0.1)' : 'rgba(16,185,129,0.1)',
                  color: trainingState === 0 ? 'var(--text-muted)' : trainingState === 1 ? 'var(--accent-cyan)' : 'var(--accent-emerald)',
                }}>
                  {trainingState === 0 ? 'Idle' : trainingState === 1 ? 'Spinning up ML Engines...' : 'Model Weights Sent'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// Slide 2: Local AI Training & DP
// ==========================================
export const LocalTrainingSlide = () => {
  const [epsilon, setEpsilon] = useState(5.0);
  
  const blurAmount = Math.max(0, (15 - epsilon) * 1.2);
  const opacityAmount = Math.max(0, (10 - epsilon) * 0.08);

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Layer 1: Hybrid Differential Privacy</h2>
      <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', maxWidth: '900px' }}>
        In our system, the <strong>Three Hospital Clients</strong> act as our primary defensive endpoints. The local security workflow operates in three strict procedural steps before communicating with the server:
      </p>

      {/* STEP-BY-STEP PRESENTATION SCRIPT MATTERS */}
      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '3rem', flexWrap: 'wrap' }}>
        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--text-muted)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 1: Raw Gradient Calculation</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>The hospital's local AI scans the patient images and uses calculus to generate <strong>gradients</strong>—the mathematically pure equations detailing exactly what it learned about the cancer patterns.</p>
        </div>
        
        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--accent-rose)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 2: Enforce Gradient Clipping</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>We programmatically chop off any gradient values that grow mathematically unbound. This enforces a strict <strong style={{color: 'var(--accent-rose)'}}>Mathematical Ceiling</strong>, stabilizing the AI to prevent nan-crashing and calibrating the upcoming noise equations.</p>
        </div>

        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--accent-cyan)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 3: Inject Differential Privacy</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>With the numbers bounded, the algorithm injects locally-calibrated <strong style={{color:'var(--accent-cyan)'}}>Gaussian Noise (static)</strong> into the matrices. This permanently obfuscates the patient's identity prior to network transmission.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem' }}>
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <div style={{ padding: '0.4rem 0.8rem', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', borderRadius: '4px', fontSize: '0.8rem', display: 'inline-block', marginBottom: '1rem', fontWeight: 'bold' }}>
            ● Endpoint Security Active on 3 Hospital Nodes
          </div>
          <h3>Privacy Control Panel</h3>
          <p style={{ marginBottom: '2rem' }}>Slide down the ε (Epsilon) value to maximize patient security.</p>
          
          <div style={{ marginBottom: '2rem' }}>
            <div className="flex-between" style={{ marginBottom: '1rem' }}>
              <span>Privacy Budget (ε): <strong>{epsilon.toFixed(1)}</strong></span>
              {epsilon < 3 ? <span style={{color: 'var(--accent-emerald)'}}>Maximum Security</span> : epsilon > 8 ? <span style={{color: 'var(--accent-rose)'}}>Low Security</span> : <span style={{color: 'var(--accent-cyan)'}}>Hybrid Balance</span>}
            </div>
            <input 
              type="range" 
              min="0.5" max="15.0" step="0.5" 
              value={epsilon} 
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          <div className="glass-pill" style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.5)' }}>
            <div style={{ marginBottom: '1rem', color: 'var(--accent-cyan)' }}><EyeOff size={24} style={{verticalAlign: 'middle', marginRight: '10px'}}/> <strong>Mathematical Guarantee</strong></div>
            <div style={{ fontFamily: 'monospace', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
              σ = Δf × √(2 ln(1.25/δ)) / ε<br/><br/>
              Standard Deviation Noise Level = {(Math.sqrt(Math.log(1.25/0.00001)) / epsilon).toFixed(4)}
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', border: epsilon > 8 ? '1px solid var(--accent-rose)' : '1px solid var(--border-glass)' }}>
          <h3 style={{ color: epsilon > 8 ? 'var(--accent-rose)' : 'var(--text-primary)' }}>Simulated Model Inversion Attack</h3>
          <p style={{ marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Visualizing what a Hacker sees if they intercept the AI equations right now.
          </p>
          
          <div style={{ 
            width: '100%', height: '280px', 
            background: 'url("/skin_scan_mockup.png")',
            backgroundSize: 'cover', backgroundPosition: 'center',
            borderRadius: '8px', border: '1px solid var(--border-glass)',
            position: 'relative', overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
              backdropFilter: `blur(${blurAmount}px)`,
              backgroundColor: `rgba(255,255,255,${opacityAmount})`,
              transition: 'all 0.2s',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              {epsilon < 2 && <Lock size={64} color="var(--accent-emerald)" />}
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', padding: '0.8rem', background: 'rgba(0,0,0,0.4)', borderRadius: '4px' }}>
            {epsilon > 8 ? (
              <span style={{ color: 'var(--accent-rose)', fontWeight: 'bold' }}>⚠️ HACKER SUCCESS: Image successfully reverse-engineered! Biomarkers exposed.</span>
            ) : epsilon < 3 ? (
              <span style={{ color: 'var(--accent-emerald)', fontWeight: 'bold' }}>🔒 HACKER DEFEATED: Attack returns pure mathematical noise.</span>
            ) : (
              <span style={{ color: 'var(--accent-cyan)' }}>Partial Reconstruction: Identity obscured but general patterns remain.</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// Slide 3: Quantum Transport
// ==========================================
export const QuantumTransportSlide = () => {
  const [hacked, setHacked] = useState(false);
  const [qStep, setQStep] = useState(0); 

  const handleNextStep = () => {
    if (qStep < 2) setQStep(qStep + 1);
  };

  const resetTarget = () => {
    setHacked(false);
    setQStep(0);
  };

  const logs = hacked ? [
      "[SYSTEM] Foreign node detected on fiber optic line.", 
      "⚠️ [WARNING] Quantum Wavefunction Collapse!", 
      "⚠️ [CHSH TEST] Score dropped to 1.41 < 2.0!", 
      "🚨 [CRITICAL ALERT] EAVESDROPPER DETECTED.", 
      "🚨 [CRITICAL ALERT] UPLINK TERMINATED."
    ] : qStep === 0 ? [
      "[SYSTEM] Awaiting Quantum Initialization..."
    ] : qStep === 1 ? [
      "[SYSTEM] Hybrid Quantum Protocol Initiated...", 
      "[SYSTEM] Executing E91 Entanglement...", 
      "[CHSH TEST] Score = 2.82 >> THRESHOLD >> PURE"
    ] : [
      "[SYSTEM] Hybrid Quantum Protocol Initiated...", 
      "[SYSTEM] Executing E91 Entanglement...", 
      "✓ [CHSH TEST] Score = 2.82 >> THRESHOLD >> PURE",
      "🔒 [CRYPTO] Wrapping Gradients in Fernet (AES-128-CBC)...",
      "🟢 [CRYPTO] Encrypted Payload securely transmitting."
    ];

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Layer 2: Quantum Entanglement Exchange (E91)</h2>
      <p style={{ fontSize: '1.1rem', marginBottom: '1.5rem', maxWidth: '900px' }}>
        Assuming the hospital's gradients are now safely scrambled with Privacy Noise, they must be transmitted to the central server. We use a simulated <strong>Quantum Fiber-Optic E91 Protocol</strong> to guarantee the network cable itself cannot be tapped.
      </p>

      {/* STEP-BY-STEP PRESENTATION SCRIPT MATTERS */}
      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '3rem', flexWrap: 'wrap' }}>
        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--text-muted)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 1: Photon Entanglement</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>The system spawns pairs of <strong>Quantum Entangled Photons</strong> (Bell States). One photon stays at the hospital, and its identical twin travels down the fiber-optic line to the central server.</p>
        </div>
        
        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--accent-cyan)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 2: The CHSH Security Test</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>Both computers measure their photons. If the mathematical <strong>CHSH Score</strong> equals ~2.82, quantum physics guarantees no hacker is looking at the cable. If a hacker looks, the score mathematically collapses below 2.0.</p>
        </div>

        <div className="glass-panel" style={{ flex: 1, padding: '1.5rem', borderLeft: '4px solid var(--accent-emerald)' }}>
          <h4 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', marginBottom: '0.8rem' }}>Step 3: Fernet (AES) Encryption Lock</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-muted)' }}>Because the CHSH test passed (&gt; 2.0), the framework uses the measured photons to mathematically generate an unbreakable <strong>Fernet AES-128 Token</strong>, locking the Medical Gradients right before they are transmitted.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        
        {/* INTERACTIVE CONTROLS */}
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Quantum Sequence Controls</h3>
          
          <button 
             onClick={handleNextStep}
             disabled={qStep >= 2 || hacked}
             style={{
               width: '100%', padding: '1rem', background: (qStep >= 2 || hacked) ? 'var(--bg-card)' : 'var(--accent-cyan)', 
               color: (qStep >= 2 || hacked) ? 'var(--text-muted)' : '#000', 
               border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: (qStep >= 2 || hacked) ? 'not-allowed' : 'pointer',
               boxShadow: (qStep < 2 && !hacked) ? '0 0 15px rgba(0,240,255,0.4)' : 'none', transition: 'all 0.3s'
             }}
          >
             {qStep === 0 ? '1. Initialize Photon Entanglement' : qStep === 1 ? '2. Verify CHSH Score & Transmit' : 'Quantum Uplink Complete'}
          </button>

          <button 
             onClick={() => setHacked(true)}
             disabled={hacked}
             style={{
               width: '100%', padding: '1rem', background: hacked ? 'var(--bg-card)' : 'var(--accent-rose)', 
               color: hacked ? 'var(--text-muted)' : '#fff', border: 'none', borderRadius: '4px', fontWeight: 'bold', 
               cursor: hacked ? 'not-allowed' : 'pointer', transition: 'all 0.3s'
             }}
          >
             ⚠️ Simulate Hacker Interception
          </button>

          <button 
             onClick={resetTarget}
             style={{
               width: '100%', padding: '0.8rem', background: 'transparent', 
               color: 'var(--text-primary)', border: '1px solid var(--border-glass)', borderRadius: '4px', cursor: 'pointer'
             }}
          >
             Reset Simulation
          </button>
        </div>

        {/* VISUALIZATION */}
        <div className="glass-panel" style={{ padding: '2rem', position: 'relative' }}>
          <div className="flex-between" style={{ marginBottom: '4rem', marginTop: '2rem' }}>
            <div style={{ textAlign: 'center', zIndex: 10 }}>
              <Database size={48} color="var(--text-primary)" />
              <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Hospital Client</div>
            </div>
            
            <div style={{ flex: 1, padding: '0 2rem', position: 'relative', zIndex: 1 }}>
              {hacked ? (
                 <div style={{ height: '4px', background: 'var(--accent-rose)', width: '100%' }} />
              ) : qStep === 0 ? (
                 <div style={{ height: '2px', background: 'var(--border-glass)', width: '100%', borderStyle: 'dashed' }} />
              ) : (
                 <div className="fiber-optic-cable"><div className={qStep === 2 ? "fiber-optic-pulse" : "fiber-optic-pulse-slow"} /></div>
              )}
              
              <div style={{ 
                position: 'absolute', top: '-35px', left: '50%', transform: 'translateX(-50%)',
                color: hacked ? 'var(--accent-rose)' : qStep === 0 ? 'var(--text-muted)' : 'var(--accent-cyan)', 
                fontWeight: 'bold', background: 'rgba(0,0,0,0.8)', padding: '0.5rem 1rem', borderRadius: '20px',
                border: hacked ? '1px solid var(--accent-rose)' : qStep > 0 ? '1px solid var(--accent-cyan)' : '1px solid var(--border-glass)'
              }}>
                {hacked ? 'X COLLAPSED (1.41)' : qStep === 0 ? 'Idle' : qStep === 1 ? 'CHSH: 2.82 (Safe)' : 'AES-128 Transmitting 🔒'}
              </div>
            </div>

            <div style={{ textAlign: 'center', zIndex: 10 }}>
              <Server size={48} color="var(--text-primary)" />
              <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Central Server</div>
            </div>
          </div>

          {qStep === 2 && !hacked && (
            <div style={{ padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid var(--accent-emerald)', borderRadius: '4px', marginBottom: '1rem', textAlign: 'center' }}>
              <div style={{ color: 'var(--accent-emerald)', fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>FERNET SECURE PAYLOAD (AES-128-CBC + HMAC-SHA256)</div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-muted)', overflowWrap: 'break-word', letterSpacing: '1px' }}>
                gAAAAABkV...8ZxL9QjP_7YkO9uK_x_T2l_mQ=
              </div>
            </div>
          )}

          <div className="glass-panel font-mono" style={{ padding: '1rem', background: '#000', color: hacked ? 'var(--accent-rose)' : 'var(--accent-emerald)', minHeight: '150px', fontSize: '0.85rem' }}>
            {logs.map((log, i) => (
              <div key={i} style={{ marginBottom: '0.5rem' }}>{log}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// Slide 4: Secure Aggregation
// ==========================================
export const SecureAggregationSlide = () => {
  const [step, setStep] = useState(0);

  const hospitals = [
    { id: 'A', status: 'Honest', norm: 1200, color: 'var(--accent-cyan)' },
    { id: 'B', status: 'Honest', norm: 1350, color: 'var(--accent-emerald)' },
    { id: 'C', status: 'POISONED DATA', norm: 8500, color: 'var(--accent-rose)' },
  ];

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Layer 3: Krum Defensive Aggregation</h2>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem', maxWidth: '800px' }}>
        When the encrypted vectors arrive, the central server runs a specialized <strong>Hybrid Krum + Trimmed Mean Protocol</strong>. It measures cosine similarities to actively isolate and mathematically purge compromised networks.
      </p>

      <div style={{ display: 'flex', gap: '2rem', marginBottom: '2rem' }}>
        {hospitals.map((h, i) => (
          <div key={h.id} className="glass-panel" style={{ 
            flex: 1, padding: '1.5rem', textAlign: 'center',
            opacity: (step > 1 && h.id === 'C') ? 0.3 : 1,
            border: (step > 1 && h.id === 'C') ? '1px solid var(--accent-rose)' : '1px solid var(--border-glass)',
            transition: 'all 0.5s'
          }}>
             <h3 style={{color: h.color}}>Network {h.id}</h3>
             <div style={{ marginTop: '1rem', fontSize: '2rem', fontWeight: 'bold' }}>{h.norm}</div>
             <div style={{ color: 'var(--text-muted)' }}>Transmission Norm</div>
             
             {step > 0 && <div style={{ 
               marginTop: '1rem', padding: '0.5rem', borderRadius: '4px',
               background: h.id === 'C' ? 'rgba(244,63,94,0.2)' : 'rgba(16,185,129,0.2)' 
             }}>
               {h.id === 'C' ? 'Malicious Deviation Flagged' : 'Honest Trajectory Verified'}
             </div>}
          </div>
        ))}
      </div>

      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center' }}>
        <button 
           onClick={() => setStep((s) => (s + 1) % 4)}
           style={{
             padding: '1rem 2rem', background: 'var(--text-primary)', 
             color: '#000', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer',
             marginBottom: '2rem'
           }}
        >
           {step === 0 ? "1. Execute Trajectory Calculations" : 
            step === 1 ? "2. Trigger Hybrid Krum Protocol" : 
            step === 2 ? "3. Finalize Safe Server Aggregation" : "Reset Terminal"}
        </button>

        <div style={{ 
          padding: '2rem', background: 'rgba(0,0,0,0.5)', borderRadius: '8px', 
          border: '1px dashed var(--accent-cyan)', fontSize: '1.5rem', fontWeight: 'bold'
        }}>
           {step === 0 ? "Awaiting Packets..." : 
            step === 1 ? "Analyzing Cosine Vector Similarities..." : 
            step >= 2 ? <span style={{color: 'var(--accent-emerald)'}}><CheckCircle style={{verticalAlign: 'bottom'}}/> Poisoned Network C Purged. Safe Hybrid Consensus Reached.</span> : ""}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// Slide 5: Final Evaluation
// ==========================================
export const FinalEvaluationSlide = () => {
  const data = [
    { round: 1, acc: 14.2, secAcc: 11.5 },
    { round: 3, acc: 35.5, secAcc: 28.1 },
    { round: 5, acc: 56.4, secAcc: 47.3 },
    { round: 7, acc: 72.1, secAcc: 61.5 },
    { round: 10, acc: 85.0, secAcc: 74.8 },
  ];

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Layer 4: Security Outputs & Performance Analysis</h2>
      
      {/* EXPLANATION ADDED PER USER REQUEST */}
      <div style={{ 
        padding: '1.5rem', background: 'rgba(255,255,255,0.05)', 
        borderLeft: '4px solid var(--accent-cyan)', borderRadius: '0 8px 8px 0', 
        marginBottom: '2rem', fontSize: '1.1rem', lineHeight: '1.6'
      }}>
        <p style={{marginBottom: '1rem'}}>
          <strong>What are we looking at?</strong> These final charts mathematically summarize why our Hybrid system is so valuable.
        </p>
        <p style={{marginBottom: '1rem'}}>
          <strong>Left Graph (Convergence):</strong> This shows how quickly the AI smartens up over 10 rounds of training. Notice the red line (an unsecured, standard AI). See how the blue line (our Hybrid Secure AI) tracks closely behind it? It proves our security layers don't break the AI's ability to learn complex cancer patterns!
        </p>
        <p>
          <strong>Right Graph (The Cost of Privacy):</strong> We must sacrifice a tiny fraction of total accuracy to guarantee that hackers can never identify our patients. This chart visually proves the literal "Cost of Privacy"—trading about 10% accuracy for 100% mathematical security against model-inversion attacks.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '3rem' }}>
        <div className="glass-panel" style={{ padding: '2rem', height: '350px' }}>
          <h3 style={{ marginBottom: '2rem' }}>Hybrid Convergence Rate</h3>
          <ResponsiveContainer width="100%" height="80%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="round" stroke="var(--text-muted)" />
              <YAxis stroke="var(--text-muted)" domain={[0, 100]} />
              <RechartsTooltip contentStyle={{ background: 'var(--bg-deep)', border: '1px solid var(--border-glass)' }} />
              <Line type="monotone" dataKey="acc" name="Standard (Unsecured)" stroke="var(--accent-rose)" strokeWidth={3} />
              <Line type="monotone" dataKey="secAcc" name="Hybrid Model (Secured)" stroke="var(--accent-cyan)" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-panel" style={{ padding: '2rem', height: '350px' }}>
          <h3 style={{ marginBottom: '2rem' }}>The 'Cost of Privacy' Trade-Off</h3>
          <ResponsiveContainer width="100%" height="80%">
            <BarChart data={[{name: 'Diagnostic Accuracy Ceiling', val1: 85.0, val2: 74.8}]}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" stroke="var(--text-muted)" />
              <YAxis stroke="var(--text-muted)" domain={[0, 100]} />
              <RechartsTooltip contentStyle={{ background: 'var(--bg-deep)', border: '1px solid var(--border-glass)' }} />
              <Bar dataKey="val1" name="Standard Model" fill="var(--accent-rose)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="val2" name="Hybrid Privacy Model" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
