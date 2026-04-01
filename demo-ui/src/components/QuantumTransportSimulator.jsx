import React, { useState, useEffect } from 'react';
import { Database, Server, Shield, AlertTriangle, Lock, Eye, CheckCircle, XCircle, Key, Terminal, Activity } from 'lucide-react';

export const QuantumTransportSimulator = () => {
  const [eveLevel, setEveLevel] = useState(0);
  const [keys, setKeys] = useState([]);

  // Mathematical Simulation of E91
  const qber = (eveLevel * 0.45).toFixed(1); // Max ~45% error
  const chsh = Math.max(0, 2.82 - (eveLevel * 0.015)).toFixed(2);
  const isSecure = parseFloat(chsh) >= 2.00;

  // Real-time Key Generation Simulation
  useEffect(() => {
    let interval;
    if (isSecure) {
      interval = setInterval(() => {
        const hex = Array.from({length: 8}, () => Math.floor(Math.random()*16).toString(16)).join('').toUpperCase();
        setKeys(prev => [...prev.slice(-4), `[SECURE] AES-256 BLOCK: ${hex}...`]);
      }, 1000);
    } else {
      setKeys(prev => {
        if (prev.length === 0 || !prev[prev.length - 1].includes("HALTED")) {
           return [...prev.slice(-4), `[FATAL] E91 WAVEFUNCTION COLLAPSED. TRANSMISSION HALTED.`, `[WARNING] EAVESDROPPER DETECTED.`];
        }
        return prev;
      });
    }
    return () => clearInterval(interval);
  }, [isSecure]);

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <style>{`
        .q-slider { -webkit-appearance: none; width: 100%; height: 8px; border-radius: 4px; background: rgba(255,255,255,0.1); outline: none; transition: all 0.2s; }
        .q-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 24px; height: 24px; border-radius: 50%; background: ${isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)'}; cursor: pointer; box-shadow: 0 0 15px ${isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)'}; border: 2px solid #000; }
        .hud-panel { background: rgba(0,0,0,0.6); border: 1px solid var(--border-glass); border-radius: 8px; padding: 1rem; }
        .metric-value { font-family: monospace; font-size: 1.8rem; font-weight: bold; text-shadow: 0 0 10px currentColor; }
        
        .fiber-core { position: absolute; top: 50%; left: 0; right: 0; height: 4px; background: rgba(255,255,255,0.1); transform: translateY(-50%); border-radius: 2px; }
        .fiber-pulse { position: absolute; height: 100%; box-shadow: 0 0 20px currentColor; border-radius: 2px; animation: zip 2s linear infinite; }
        @keyframes zip { 0% { left: 0; width: 0%; opacity: 0; } 10% { opacity: 1; } 50% { width: 30%; left: 35%; opacity: 1; } 90% { opacity: 1; } 100% { left: 100%; width: 0%; opacity: 0; } }
        
        .pulse-warn { animation: warning 1s infinite alternate; }
        @keyframes warning { from { opacity: 0.5; box-shadow: 0 0 10px var(--accent-rose); } to { opacity: 1; box-shadow: 0 0 30px var(--accent-rose); } }
      `}</style>
      
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>E91 Quantum Encryption Sandbox</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>Actively simulate eavesdropper interference on the fiber-optic Bell State photons.</p>
      </div>

      <div style={{ display: 'flex', gap: '2rem', marginBottom: '3rem' }}>
        
        {/* Left Control Panel: Eavesdropper Simulation */}
        <div className="glass-panel" style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', border: isSecure ? '1px solid var(--border-glass)' : '1px solid var(--accent-rose)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)' }}>
            <Eye size={24} />
            <h3 style={{ margin: 0, color: 'currentColor' }}>Hacker Interference Control</h3>
          </div>
          
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Increase the slider to simulate an attacker (Eve) attempting to measure the entangled photons. According to the No-Cloning Theorem, this alters their quantum state.
          </p>

          <div style={{ padding: '2rem 1rem', background: 'rgba(0,0,0,0.4)', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
             <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', fontWeight: 'bold' }}>
               <span style={{ color: 'var(--accent-emerald)' }}>Clean Line (0%)</span>
               <span style={{ color: 'var(--accent-rose)' }}>Active Wiretap (100%)</span>
             </div>
             <input 
                type="range" 
                min="0" max="100" 
                value={eveLevel} 
                onChange={(e) => setEveLevel(e.target.value)}
                className="q-slider"
             />
             <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '1.5rem', fontWeight: 'bold', color: isSecure ? 'var(--text-primary)' : 'var(--accent-rose)' }}>
                Interference Noise: {eveLevel}%
             </div>
          </div>
        </div>

        {/* Right Panel: Mathematical Telemetry */}
        <div className="glass-panel" style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={24} color="var(--accent-cyan)" /> Live Telemetry Dashboard
          </h3>
          
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
             
             {/* CHSH Score */}
             <div className="hud-panel" style={{ flex: 1, borderColor: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>CHSH Inequality S-Value</div>
                <div className="metric-value" style={{ color: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)' }}>
                  {chsh}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  {isSecure ? '≥ 2.0 (Quantum Entangled)' : '< 2.0 (Wavefunction Collapsed)'}
                </div>
             </div>

             {/* QBER Score */}
             <div className="hud-panel" style={{ flex: 1, borderColor: isSecure ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Quantum Bit Error Rate</div>
                <div className="metric-value" style={{ color: isSecure ? (qber < 5 ? 'var(--accent-emerald)' : 'var(--accent-purple)') : 'var(--accent-rose)' }}>
                  {qber}%
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  {isSecure ? 'Within tolerable limits' : 'Unacceptable noise detected'}
                </div>
             </div>
          </div>

          {/* Alert Banner */}
          <div style={{ 
            marginTop: 'auto', padding: '1rem', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '1rem',
            background: isSecure ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255, 51, 102, 0.1)',
            border: isSecure ? '1px solid var(--accent-emerald)' : '1px solid var(--accent-rose)',
            color: isSecure ? 'var(--accent-emerald)' : 'var(--accent-rose)'
          }}>
            {isSecure ? <CheckCircle size={28} /> : <XCircle size={28} />}
            <div style={{ fontWeight: 'bold', fontSize: '1.1rem' }}>
               {isSecure ? 'SYSTEM SECURE: Payload generation active.' : 'SYSTEM BREACHED: Protocol aborted instantly.'}
            </div>
          </div>
        </div>

      </div>

      {/* Main Connection Visualizer */}
      <h3 style={{ color: 'var(--text-muted)', fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '2px', textAlign: 'center', marginBottom: '1.5rem' }}>Physical Fiber-Optic Layer</h3>
      <div className={`glass-panel ${!isSecure ? 'pulse-warn' : ''}`} style={{ padding: '2rem', position: 'relative', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: isSecure ? '1px solid var(--border-glass)' : '2px solid var(--accent-rose)' }}>
        
        {/* Client */}
        <div style={{ textAlign: 'center', zIndex: 10 }}>
          <Database size={48} color={isSecure ? "var(--text-primary)" : "var(--accent-rose)"} />
          <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Hospital AI</div>
        </div>

        {/* The Wire */}
        <div style={{ flex: 1, margin: '0 2rem', position: 'relative', height: '40px' }}>
           <div className="fiber-core" style={{ background: isSecure ? 'rgba(0, 240, 255, 0.2)' : 'rgba(255, 51, 102, 0.2)' }}></div>
           
           {isSecure && (
             <>
               <div className="fiber-pulse" style={{ color: 'var(--accent-cyan)', background: 'var(--accent-cyan)', left: 0 }}></div>
               <div className="fiber-pulse" style={{ color: '#fff', background: '#fff', animationDelay: '1s' }}></div>
               <div style={{ position: 'absolute', top: '-15px', width: '100%', textAlign: 'center', color: 'var(--accent-cyan)', fontSize: '0.8rem', fontWeight: 'bold' }}>PHOTONS IN SUPERPOSITION</div>
             </>
           )}
           
           {!isSecure && (
             <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: '#000', padding: '0.5rem 1rem', border: '1px solid var(--accent-rose)', color: 'var(--accent-rose)', fontWeight: 'bold', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
               <AlertTriangle size={16} /> CONNECTION SEVERED
             </div>
           )}
        </div>

        {/* Server */}
        <div style={{ textAlign: 'center', zIndex: 10 }}>
          <Server size={48} color={isSecure ? "var(--text-primary)" : "var(--accent-rose)"} />
          <div style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>Central Krum Node</div>
        </div>

      </div>

      {/* Encryption Key Generator Terminal */}
      <div style={{ marginTop: '2rem', background: '#0A0A0A', border: '1px solid #333', borderRadius: '8px', padding: '1.5rem', fontFamily: 'monospace', minHeight: '160px', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, background: '#1A1A1A', padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#888', fontSize: '0.8rem' }}>
          <Terminal size={14} /> AES-256 Symmetric Key Generator
        </div>
        <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
           {keys.map((k, i) => (
             <div key={i} style={{ color: k.includes("FATAL") || k.includes("WARNING") ? 'var(--accent-rose)' : 'var(--accent-emerald)', opacity: 0.8 + (i * 0.05) }}>
               <span style={{ color: '#555' }}>[{new Date().toISOString().split('T')[1].replace('Z', '')}]</span> {k}
             </div>
           ))}
        </div>
      </div>

    </div>
  );
};
