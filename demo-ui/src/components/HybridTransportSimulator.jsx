import React, { useState, useEffect } from 'react';
import { Database, Server, Shield, AlertTriangle, Lock, Eye, CheckCircle, XCircle, Key, Terminal, Activity, Wifi, WifiOff } from 'lucide-react';

export const QuantumTransportSimulator = () => {
  const [eveLevel, setEveLevel] = useState(0);
  const [keys, setKeys] = useState([]);

  const qber = (eveLevel * 0.45).toFixed(1);
  const chsh = Math.max(0, 2.82 - (eveLevel * 0.015)).toFixed(2);
  const isSecure = parseFloat(chsh) >= 2.00;

  useEffect(() => {
    let interval;
    if (isSecure) {
      interval = setInterval(() => {
        const hex = Array.from({length: 8}, () => Math.floor(Math.random()*16).toString(16)).join('').toUpperCase();
        setKeys(prev => [...prev.slice(-4), `[SECURE] AES-256 BLOCK: ${hex}...`]);
      }, 1000);
    } else {
      setTimeout(() => {
        setKeys(prev => {
          if (prev.length === 0 || !prev[prev.length - 1].includes("HALTED")) {
             return [...prev.slice(-4), `[FATAL] E91 WAVEFUNCTION COLLAPSED. TRANSMISSION HALTED.`, `[WARNING] EAVESDROPPER DETECTED.`];
          }
          return prev;
        });
      }, 0);
    }
    return () => clearInterval(interval);
  }, [isSecure]);

  return (
    <div className="animate-slide-up">
      <style>{`
        .fiber-core { position: absolute; top: 50%; left: 0; right: 0; height: 2px; background: rgba(255,255,255,0.06); transform: translateY(-50%); }
        .photon-packet {
          position: absolute;
          top: 50%;
          transform: translateY(-50%) translateX(-50%);
          animation: photonTravel 3s cubic-bezier(0.4, 0, 0.2, 1) infinite;
          z-index: 5;
        }
        @keyframes photonTravel {
          0% { left: 0%; opacity: 0; scale: 0.8; }
          15% { opacity: 1; scale: 1; }
          85% { opacity: 1; scale: 1; }
          100% { left: 100%; opacity: 0; scale: 0.8; }
        }
        .pulse-warn { animation: warning 1s infinite alternate; }
        @keyframes warning { from { opacity: 0.6; box-shadow: 0 0 8px var(--accent-rose); } to { opacity: 1; box-shadow: 0 0 24px var(--accent-rose); } }
      `}</style>
      
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>E91 Quantum Encryption Sandbox</h2>
        <p style={{ fontSize: '1rem', maxWidth: '700px' }}>Simulate eavesdropper interference on the fiber-optic Bell State photon channel in real time.</p>
      </div>

      {/* Two Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* LEFT: Eavesdropper Control */}
        <div className="glass-panel" style={{ 
          padding: '1.5rem', 
          border: isSecure ? '1px solid var(--border-glass)' : '1px solid rgba(244, 63, 94, 0.4)',
          transition: 'border-color 0.4s'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)' }}>
            <Eye size={20} />
            <h3 style={{ margin: 0, color: 'inherit', fontSize: '1rem' }}>Hacker Interference Control</h3>
          </div>
          
          <p style={{ fontSize: '0.85rem', marginBottom: '1.25rem', color: 'var(--text-muted)' }}>
            Increase the slider to simulate an attacker (Eve) measuring entangled photons. The No-Cloning Theorem ensures this alters their quantum state.
          </p>

          <div style={{ padding: '1.25rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem', fontSize: '0.75rem', fontWeight: 600 }}>
              <span style={{ color: 'var(--accent-emerald)' }}>Clean (0%)</span>
              <span style={{ color: 'var(--accent-rose)' }}>Active Wiretap (100%)</span>
            </div>
            <input 
              type="range" 
              min="0" max="100" 
              value={eveLevel} 
              onChange={(e) => setEveLevel(e.target.value)}
              style={{
                '--thumb-color': isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)',
              }}
            />
            <div style={{ 
              textAlign: 'center', marginTop: '1rem', 
              fontSize: '1.5rem', fontWeight: 700, 
              fontFamily: "'JetBrains Mono', monospace",
              color: isSecure ? 'var(--text-primary)' : 'var(--accent-rose)' 
            }}>
              {eveLevel}%
            </div>
          </div>
        </div>

        {/* RIGHT: Telemetry */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--accent-cyan)" />
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Live Telemetry</h3>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            {/* CHSH */}
            <div style={{ 
              background: 'rgba(0,0,0,0.4)', borderRadius: 'var(--radius-md)', padding: '1rem',
              border: `1px solid ${isSecure ? 'rgba(0, 240, 255, 0.2)' : 'rgba(244, 63, 94, 0.3)'}`,
              transition: 'border-color 0.3s'
            }}>
              <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>CHSH S-Value</div>
              <div style={{ 
                fontFamily: "'JetBrains Mono', monospace", fontSize: '2rem', fontWeight: 700,
                color: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)',
                textShadow: `0 0 16px ${isSecure ? 'rgba(0,240,255,0.4)' : 'rgba(244,63,94,0.4)'}`,
              }}>{chsh}</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                {isSecure ? '≥ 2.0 — Entangled' : '< 2.0 — Collapsed'}
              </div>
            </div>
            
            {/* QBER */}
            <div style={{ 
              background: 'rgba(0,0,0,0.4)', borderRadius: 'var(--radius-md)', padding: '1rem',
              border: `1px solid ${isSecure ? 'rgba(16,185,129,0.2)' : 'rgba(244,63,94,0.3)'}`,
              transition: 'border-color 0.3s'
            }}>
              <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>Bit Error Rate</div>
              <div style={{ 
                fontFamily: "'JetBrains Mono', monospace", fontSize: '2rem', fontWeight: 700,
                color: isSecure ? (qber < 5 ? 'var(--accent-emerald)' : 'var(--accent-purple)') : 'var(--accent-rose)',
                textShadow: `0 0 16px ${isSecure ? 'rgba(16,185,129,0.4)' : 'rgba(244,63,94,0.4)'}`,
              }}>{qber}%</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                {isSecure ? 'Within limits' : 'Unacceptable noise'}
              </div>
            </div>
          </div>

          {/* Status Banner */}
          <div style={{ 
            marginTop: 'auto', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', 
            display: 'flex', alignItems: 'center', gap: '0.75rem',
            background: isSecure ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
            border: isSecure ? '1px solid rgba(16,185,129,0.25)' : '1px solid rgba(244,63,94,0.25)',
            color: isSecure ? 'var(--accent-emerald)' : 'var(--accent-rose)',
            transition: 'all 0.3s',
          }}>
            {isSecure ? <CheckCircle size={22} /> : <XCircle size={22} />}
            <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
              {isSecure ? 'SYSTEM SECURE — Payload active' : 'SYSTEM BREACHED — Protocol aborted'}
            </div>
          </div>
        </div>
      </div>

      {/* Fiber Optic Visualizer */}
      <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-dim)', textAlign: 'center', marginBottom: '0.75rem', fontWeight: 600 }}>
        Physical Fiber-Optic Layer
      </div>
      <div className={`glass-panel ${!isSecure ? 'pulse-warn' : ''}`} style={{ 
        padding: '1.5rem 2rem', position: 'relative', 
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
        border: isSecure ? '1px solid var(--border-glass)' : '1px solid rgba(244,63,94,0.4)',
        transition: 'border-color 0.3s',
      }}>
        {/* Client */}
        <div style={{ textAlign: 'center', zIndex: 10, minWidth: '80px' }}>
          <Database size={40} color={isSecure ? "var(--accent-cyan)" : "var(--accent-rose)"} />
          <div style={{ marginTop: '0.4rem', fontWeight: 600, fontSize: '0.8rem' }}>Hospital AI</div>
        </div>

        {/* Wire */}
        <div style={{ flex: 1, margin: '0 1.5rem', position: 'relative', height: '40px' }}>
          <div className="fiber-core" style={{ background: isSecure ? 'rgba(0, 240, 255, 0.2)' : 'rgba(244, 63, 94, 0.2)' }} />
          {isSecure && (
            <>
              {/* AES Encrypted Packet */}
              <div className="photon-packet" style={{ animationDelay: '0s' }}>
                 <div style={{ padding: '0.25rem 0.75rem', background: 'rgba(0,240,255,0.15)', border: '1px solid rgba(0,240,255,0.5)', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.65rem', fontWeight: 700, color: 'var(--accent-cyan)', boxShadow: '0 0 16px rgba(0,240,255,0.3)', backdropFilter: 'blur(4px)' }}>
                    <Lock size={12} /> SECURE PAYLOAD
                 </div>
              </div>
              
              {/* Quantum Check Packet */}
              <div className="photon-packet" style={{ animationDelay: '1.5s' }}>
                 <div style={{ padding: '0.25rem 0.75rem', background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.5)', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.65rem', fontWeight: 700, color: 'var(--accent-emerald)', boxShadow: '0 0 16px rgba(16,185,129,0.3)', backdropFilter: 'blur(4px)' }}>
                    <Shield size={12} /> CHSH VERIFIED
                 </div>
              </div>

              <div style={{ position: 'absolute', top: '-18px', width: '100%', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em' }}>
                FIBER OPTIC LINK ACTIVE
              </div>
            </>
          )}
          {!isSecure && (
            <div style={{ 
              position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', 
              background: 'rgba(0,0,0,0.8)', padding: '0.4rem 0.8rem', 
              border: '1px solid var(--accent-rose)', color: 'var(--accent-rose)', fontWeight: 700, 
              borderRadius: '16px', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem',
              boxShadow: '0 0 12px rgba(244,63,94,0.3)',
            }}>
              <WifiOff size={14} /> CONNECTION SEVERED
            </div>
          )}
        </div>

        {/* Server */}
        <div style={{ textAlign: 'center', zIndex: 10, minWidth: '80px' }}>
          <Server size={40} color={isSecure ? "var(--accent-emerald)" : "var(--accent-rose)"} />
          <div style={{ marginTop: '0.4rem', fontWeight: 600, fontSize: '0.8rem' }}>Krum Node</div>
        </div>
      </div>

      {/* Terminal */}
      <div className="glass-panel font-mono" style={{ 
        marginTop: '1.5rem', padding: '1.25rem', 
        background: 'rgba(5, 5, 12, 0.85)', 
        border: isSecure ? '1px solid rgba(0,240,255,0.15)' : '1px solid rgba(244,63,94,0.2)',
        borderRadius: 'var(--radius-md)',
        boxShadow: isSecure ? 'inset 0 0 30px rgba(0,240,255,0.03)' : 'inset 0 0 30px rgba(244,63,94,0.05)',
        transition: 'all 0.3s',
      }}>
        <div style={{ 
          display: 'flex', alignItems: 'center', gap: '0.5rem', 
          color: isSecure ? 'var(--accent-cyan)' : 'var(--accent-rose)', 
          fontSize: '0.75rem', fontWeight: 700, marginBottom: '1rem',
          paddingBottom: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.06)'
        }}>
          <Terminal size={14} />
          AES-256 Symmetric Key Generator
          <div style={{ marginLeft: 'auto', width: '8px', height: '8px', borderRadius: '50%', background: isSecure ? 'var(--accent-emerald)' : 'var(--accent-rose)', boxShadow: `0 0 6px ${isSecure ? 'var(--accent-emerald)' : 'var(--accent-rose)'}` }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.8rem' }}>
          {keys.map((k, i) => (
            <div key={i} style={{ color: k.includes("FATAL") || k.includes("WARNING") ? 'var(--accent-rose)' : 'var(--accent-emerald)', opacity: 0.7 + (i * 0.08) }}>
              <span style={{ color: 'var(--text-dim)' }}>[{new Date().toISOString().split('T')[1].replace('Z', '')}]</span> {k}
            </div>
          ))}
          {keys.length === 0 && <div style={{ color: 'var(--text-dim)' }}>Awaiting initialization...</div>}
        </div>
      </div>
    </div>
  );
};
