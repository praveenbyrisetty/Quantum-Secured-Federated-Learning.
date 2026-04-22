import React, { useState, useEffect, useRef } from 'react';
import { Shield, Zap, Server, CheckCircle, AlertTriangle, Lock, Unlock, EyeOff, Key, Database, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';


// ==========================================
// Slide 1: Architecture & CNN Overview
// ==========================================
export const DataDistributionSlide = () => {
  const [trainingState, setTrainingState] = useState(0);

  const executeTrainingCycle = () => {
    setTrainingState(1);
    setTimeout(() => setTrainingState(2), 2000); 
    setTimeout(() => setTrainingState(3), 5000); 
  };

  const hospitals = [
    { id: 'A', color: 'var(--accent-cyan)' },
    { id: 'B', color: 'var(--accent-emerald)' },
    { id: 'C', color: 'var(--accent-purple)' },
  ];

  return (
    <div className="animate-slide-up">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Core Architecture & AI Brain</h2>
        <p style={{ fontSize: '1rem', maxWidth: '750px' }}>
          Our Federated Learning (FL) setup distributes a customized Convolutional Neural Network across multiple simulated hospitals, strictly preserving patient privacy without data centralization.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2rem' }}>
        
        {/* LEFT COLUMN: CNN Architecture Details */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="glass-panel" style={{ padding: '1.5rem', flex: 1, border: '1px solid var(--border-glass)' }}>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🧠 Custom CNN Architecture
            </h3>
            <p style={{ fontSize: '0.85rem', marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
              Designed specifically for complex dermoscopic imagery. The model takes a raw skin image and passes it through three sequential convolutional blocks to extract geometric features.
            </p>

            {/* AI Flow Visualization */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
              
              {/* Input Image */}
              <div style={{ padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', textAlign: 'center', border: '1px dashed var(--text-dim)' }}>
                <div style={{ width: '45px', height: '45px', background: 'linear-gradient(135deg, rgba(140,80,50,0.8), rgba(80,30,20,0.8))', borderRadius: '4px', margin: '0 auto 0.5rem' }} />
                <div style={{ fontSize: '0.65rem', fontWeight: 600 }}>128x128 Px<br/>Skin Image</div>
              </div>

              {/* Convolution Blocks */}
              <div style={{ display: 'flex', flex: 1, gap: '0.5rem', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', background: 'rgba(0,240,255,0.03)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(0,240,255,0.1)' }}>
                
                {/* Block 1 */}
                <div style={{ textAlign: 'center' }}>
                  <div style={{ width: '32px', height: '40px', background: 'rgba(0,240,255,0.2)', border: '1px solid var(--accent-cyan)', borderRadius: '4px', margin: '0 auto 0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.6rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>32</div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>Block 1</div>
                  <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)' }}>Edges/Colors</div>
                </div>

                <div style={{ flex: 1, height: '2px', background: 'var(--border-glass)' }} />

                {/* Block 2 */}
                <div style={{ textAlign: 'center' }}>
                  <div style={{ width: '40px', height: '48px', background: 'rgba(16,185,129,0.2)', border: '1px solid var(--accent-emerald)', borderRadius: '4px', margin: '0 auto 0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.6rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>64</div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>Block 2</div>
                  <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)' }}>Shapes</div>
                </div>

                <div style={{ flex: 1, height: '2px', background: 'var(--border-glass)' }} />

                {/* Block 3 */}
                <div style={{ textAlign: 'center' }}>
                  <div style={{ width: '48px', height: '56px', background: 'rgba(168,85,247,0.2)', border: '1px solid var(--accent-purple)', borderRadius: '4px', margin: '0 auto 0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.65rem', fontWeight: 700, color: 'var(--accent-purple)' }}>128</div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>Block 3</div>
                  <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)' }}>Complex Params</div>
                </div>

              </div>
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)' }}>
              <strong>The Eyes (Convolutional Layers):</strong> The AI expands the feature channels exponentially at each block, identifying complex lesion borders and diagnostic patterns across 128 deep feature channels before aggregation.
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Network Distribution */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          
          <button 
            className="btn-primary" onClick={executeTrainingCycle} disabled={trainingState > 0}
            style={{ padding: '0.7rem 1.5rem', fontSize: '0.9rem', marginBottom: '1.5rem', width: '100%', justifyContent: 'center' }}
          >
            {trainingState === 0 ? 'Deploy Custom CNN to Hospitals' : 
             trainingState === 1 ? '↓ Broadcasting Network Weights…' : 
             trainingState === 2 ? '⚙️ Hospitals Crunching Local Data…' : 
             '✓ Encrypted Weights Transmitted'}
          </button>

          {/* Central Server */}
          <div className="glass-panel" style={{ 
            padding: '1rem', width: '100%', textAlign: 'center', zIndex: 10,
            border: trainingState === 3 ? '1px solid rgba(0,240,255,0.3)' : '1px solid var(--border-glass)',
            boxShadow: trainingState === 3 ? '0 0 30px rgba(0,240,255,0.1)' : 'none', transition: 'all 0.5s'
          }}>
            <Server size={28} color={trainingState === 3 ? 'var(--accent-cyan)' : 'var(--accent-purple)'} style={{ margin: '0 auto' }} />
            <h3 style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>Aggregation Server</h3>
            <div style={{ color: trainingState === 3 ? 'var(--accent-cyan)' : 'var(--text-muted)', fontSize: '0.7rem' }}>
              {trainingState === 0 ? 'Holding CNN Model v1.0' :
               trainingState === 1 ? 'Distributing 128-Channel CNN…' :
               trainingState === 2 ? 'Awaiting Computations…' : 'Models Received for Defense'}
            </div>
          </div>

          {/* Connection Lines */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', width: '80%', height: '40px', position: 'relative', zIndex: 1 }}>
            <style>{`
              @keyframes slideUpLine { 0% { height: 0; opacity: 0; } 100% { height: 100%; opacity: 1; } }
              @keyframes slideDownLine { 0% { height: 0; top: 0; opacity: 0; } 100% { height: 100%; top: 0; opacity: 1; } }
            `}</style>
            
            {/* Left Branch */}
            <div style={{ position: 'relative' }}>
              <div style={{ position: 'absolute', top: '20px', left: '50%', right: '-0.5rem', borderTop: '1px dashed var(--border-glass)' }} />
              <div style={{ position: 'absolute', top: '20px', left: '50%', bottom: 0, borderLeft: '1px dashed var(--border-glass)' }} />
              {trainingState === 1 && <div style={{ position: 'absolute', top: '20px', left: '50%', width: '2px', height: '20px', background: 'linear-gradient(to bottom, transparent, var(--accent-cyan))', animation: 'slideDownLine 1s infinite' }} />}
              {trainingState === 3 && <div style={{ position: 'absolute', bottom: 0, left: '50%', width: '2px', height: '20px', background: 'linear-gradient(to top, var(--accent-cyan), transparent)', animation: 'slideUpLine 1s infinite' }} />}
            </div>

            {/* Middle Branch */}
            <div style={{ position: 'relative' }}>
              <div style={{ position: 'absolute', top: '20px', left: '-0.5rem', right: '-0.5rem', borderTop: '1px dashed var(--border-glass)' }} />
              <div style={{ position: 'absolute', top: 0, left: '50%', bottom: 0, borderLeft: '1px dashed var(--border-glass)' }} />
              {trainingState === 1 && <div style={{ position: 'absolute', top: 0, left: '50%', width: '2px', height: '100%', background: 'linear-gradient(to bottom, transparent, var(--accent-emerald))', animation: 'slideDownLine 1s infinite 0.2s' }} />}
              {trainingState === 3 && <div style={{ position: 'absolute', bottom: 0, left: '50%', width: '2px', height: '100%', background: 'linear-gradient(to top, var(--accent-emerald), transparent)', animation: 'slideUpLine 1s infinite 0.2s' }} />}
            </div>

            {/* Right Branch */}
            <div style={{ position: 'relative' }}>
              <div style={{ position: 'absolute', top: '20px', left: '-0.5rem', right: '50%', borderTop: '1px dashed var(--border-glass)' }} />
              <div style={{ position: 'absolute', top: '20px', left: '50%', bottom: 0, borderLeft: '1px dashed var(--border-glass)' }} />
              {trainingState === 1 && <div style={{ position: 'absolute', top: '20px', left: '50%', width: '2px', height: '20px', background: 'linear-gradient(to bottom, transparent, var(--accent-purple))', animation: 'slideDownLine 1s infinite 0.4s' }} />}
              {trainingState === 3 && <div style={{ position: 'absolute', bottom: 0, left: '50%', width: '2px', height: '20px', background: 'linear-gradient(to top, var(--accent-purple), transparent)', animation: 'slideUpLine 1s infinite 0.4s' }} />}
            </div>
          </div>

          {/* Hospital Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', width: '100%', zIndex: 10 }}>
            {hospitals.map((h) => (
              <div key={h.id} className="glass-panel" style={{ 
                padding: '0.8rem', textAlign: 'center', borderTop: `3px solid ${h.color}`,
                boxShadow: trainingState === 2 ? `0 0 20px ${h.color}33` : undefined,
                transform: trainingState === 2 ? 'translateY(-2px)' : 'none'
              }}>
                <h4 style={{ marginBottom: '0.5rem', fontSize: '0.8rem' }}>Client {h.id}</h4>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '0.4rem', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', marginBottom: '0.5rem' }}>
                  <div style={{ textAlign: 'center', fontSize: '0.65rem' }}>
                    <div style={{ fontWeight: 600 }}>HAM10000 Data</div>
                  </div>
                </div>
                <div style={{ 
                  padding: '0.3rem', borderRadius: 'var(--radius-sm)', fontSize: '0.65rem', fontWeight: 600,
                  background: trainingState === 0 ? 'rgba(255,255,255,0.03)' : trainingState === 1 ? 'rgba(0,240,255,0.06)' : trainingState === 2 ? 'rgba(168,85,247,0.08)' : 'rgba(16,185,129,0.06)',
                  color: trainingState === 0 ? 'var(--text-dim)' : trainingState === 1 ? 'var(--accent-cyan)' : trainingState === 2 ? 'var(--accent-purple)' : 'var(--accent-emerald)',
                }}>
                  {trainingState === 0 ? 'Idle' : trainingState === 1 ? 'Recv' : trainingState === 2 ? 'Train' : 'Done'}
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
// Slide 2: Differential Privacy & Clipping
// ==========================================
export const LocalTrainingSlide = () => {
  const [epsilon, setEpsilon] = useState(15.0); // Default to match backend
  const [norm, setNorm] = useState(21000); // 21000 represents "DYNAMIC"
  
  // Recalibrate blur math for epsilon between 0 and 50
  const blurAmount = Math.max(0, (30 - epsilon) * 0.8);
  const opacityAmount = Math.max(0, (25 - epsilon) * 0.04);

  const steps = [
    { title: 'Raw Gradient Calculation',  desc: 'Hospital AI scans patient images and generates gradients — the mathematical equations detailing what it learned about cancer patterns.', borderColor: 'var(--text-dim)' },
    { title: 'Enforce Gradient Clipping', desc: 'Programmatically chop off any gradient values that grow unbound. This enforces a strict mathematical ceiling and calibrates the noise.', borderColor: 'var(--accent-rose)' },
    { title: 'Inject Differential Privacy', desc: 'With numbers bounded, inject locally-calibrated Gaussian Noise into the matrices. This permanently obfuscates the patient\'s identity.', borderColor: 'var(--accent-cyan)' },
  ];

  return (
    <div className="animate-slide-up">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Quantum E91 Differential Privacy</h2>
        <p style={{ fontSize: '1rem', maxWidth: '750px' }}>
          The <strong style={{ color: 'var(--text-primary)' }}>Three Hospital Clients</strong> act as our primary defensive endpoints. The local security workflow operates in three strict procedural steps before communicating with the server.
        </p>
      </div>

      {/* 3 Step Explainers */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        {steps.map((step, i) => (
          <div key={i} className="glass-panel" style={{ padding: '1.25rem', borderLeft: `3px solid ${step.borderColor}` }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '0.6rem' }}>Step {i + 1}: {step.title}</h4>
            <p style={{ fontSize: '0.82rem', margin: 0 }}>{step.desc}</p>
          </div>
        ))}
      </div>

      {/* Control + Visualization */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
        
        {/* Privacy Control Panel */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ 
            display: 'inline-flex', padding: '0.3rem 0.6rem', 
            background: 'rgba(16,185,129,0.08)', color: 'var(--accent-emerald)', 
            borderRadius: 'var(--radius-pill)', fontSize: '0.7rem', fontWeight: 600, marginBottom: '1rem' 
          }}>
            ● Endpoint Security Active on 3 Nodes
          </div>
          <h3 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>DP & Clipping Parameters</h3>
          <p style={{ fontSize: '0.85rem', marginBottom: '1.5rem' }}>Configure the mathematical bounds for patient security.</p>
          
          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span>Privacy Budget (ε): <strong>{epsilon.toFixed(1)}</strong></span>
              {epsilon <= 5.0 ? <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>Maximum Security</span> : 
               epsilon > 25.0 ? <span style={{ color: 'var(--accent-rose)', fontWeight: 600 }}>Low Security</span> : 
               <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Balanced Trade-off</span>}
            </div>
            <input type="range" min="0.5" max="50.0" step="0.5" value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))} />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span>Norm Clipping Threshold (C): <strong>{norm === 21000 ? 'DYNAMIC' : norm.toLocaleString()}</strong></span>
               {norm === 21000 ? <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>Auto (Mean × 1.5)</span> : 
               norm < 8000 ? <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Strict Ceiling</span> : 
               <span style={{ color: 'var(--accent-rose)', fontWeight: 600 }}>Loose Bounds</span>}
            </div>
            <input type="range" min="1000" max="21000" step="1000" value={norm} onChange={(e) => setNorm(parseInt(e.target.value))} />
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.4rem', lineHeight: 1.4 }}>
              Limits the maximum magnitude (L2 norm) of weight updates. Set to max for DYNAMIC variance tracking.
            </div>
          </div>

          <div style={{ padding: '1.25rem', background: 'rgba(0,0,0,0.35)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)' }}>
            <div style={{ marginBottom: '0.75rem', color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}>
              <EyeOff size={18} /> <strong>Mathematical Guarantee</strong>
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              σ = Δf × √(2 ln(1.25/δ)) / ε<br/><br/>
              Noise Level = <span style={{ color: 'var(--accent-cyan)' }}>{(Math.sqrt(Math.log(1.25/0.00001)) / epsilon).toFixed(4)}</span>
            </div>
          </div>
        </div>


          
          {/* Active Defense Visualizer */}
          <div className="glass-panel" style={{ 
            padding: '1.25rem', textAlign: 'center', 
            border: '1px solid var(--border-glass)',
            display: 'flex', flexDirection: 'column'
          }}>
            <h3 style={{ fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '0.4rem' }}>Active Defense Visualizer</h3>
            <p style={{ fontSize: '0.75rem', marginBottom: '1rem' }}>Observing Gradient Bounds & Privacy Noise Injection in real-time.</p>
            
            <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', position: 'relative', padding: '0.5rem 1rem 0', minHeight: '160px', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)' }}>
              {(() => {
                const actualNorm = norm === 21000 ? 13350 : norm; 
                return (
                  <>
                    <div style={{
                      position: 'absolute', bottom: `${Math.min((actualNorm / 20000) * 100, 100)}%`, left: 0, right: 0,
                      borderTop: '2px dashed var(--accent-rose)', transition: 'bottom 0.4s', zIndex: 10
                    }}>
                       <div style={{ position: 'absolute', right: '10px', top: '-18px', fontSize: '0.65rem', color: 'var(--accent-rose)', fontWeight: 700, backgroundColor: 'rgba(0,0,0,0.6)', padding: '0 4px', borderRadius: '4px' }}>
                         Ceiling: {actualNorm.toFixed(0)}
                       </div>
                    </div>

                    {[3000, 7500, 18000, 4000, 12000].map((val, i) => {
                      const heightPct = Math.min((val / 20000) * 100, 100);
                      const isClipped = val > actualNorm;
                      const clippedHeight = Math.min((actualNorm / 20000) * 100, 100);

                      return (
                        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '35%', maxWidth: '35px', gap: '0.4rem' }}>
                           <div style={{ position: 'relative', width: '100%', height: '130px', display: 'flex', alignItems: 'flex-end' }}>
                              {isClipped && (
                                <div style={{ 
                                  position: 'absolute', bottom: 0, width: '100%', height: `${heightPct}%`, 
                                  background: 'repeating-linear-gradient(45deg, rgba(244,63,94,0.15), rgba(244,63,94,0.15) 3px, transparent 3px, transparent 6px)',
                                  border: '1px solid rgba(244,63,94,0.4)', borderBottom: 'none', borderTopLeftRadius: '3px', borderTopRightRadius: '3px', zIndex: 5
                                }} />
                              )}
                              <div style={{ 
                                width: '100%', height: `${isClipped ? clippedHeight : heightPct}%`,
                                background: 'linear-gradient(to top, rgba(0,240,255,0.2), rgba(0,240,255,0.8))',
                                borderTopLeftRadius: '3px', borderTopRightRadius: '3px', boxShadow: `0 0 12px rgba(0,240,255,0.2)`,
                                transition: 'height 0.4s', zIndex: 10, position: 'relative', overflow: 'hidden'
                              }}>
                                <div style={{ 
                                  position: 'absolute', inset: 0, mixBlendMode: 'overlay',
                                  background: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='${opacityAmount * 1.5}'/%3E%3C/svg%3E")`
                                }} />
                              </div>
                           </div>
                           <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', fontWeight: 600 }}>G{i+1}</div>
                        </div>
                      );
                    })}
                  </>
                );
              })()}
            </div>

            <div style={{ marginTop: '1rem', padding: '0.5rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', fontWeight: 600 }}>
              {norm === 21000 ? <span style={{ color: 'var(--accent-emerald)' }}>DYNAMIC MULTIPLIER: 1.5× Mean.</span> : 
               norm <= 5000 ? <span style={{ color: 'var(--accent-cyan)' }}>STRICT BOUNDING: Outliers chopped.</span> : 
               <span style={{ color: 'var(--accent-rose)' }}>LOOSE BOUNDS: Allows high variance.</span>}
            </div>
          </div>

          {/* Model Inversion Attack Visualizer */}
          <div className="glass-panel" style={{ 
            padding: '1.25rem', textAlign: 'center', 
            border: epsilon > 25.0 ? '1px solid rgba(244,63,94,0.3)' : '1px solid var(--border-glass)',
            transition: 'border-color 0.3s'
          }}>
            <h3 style={{ fontSize: '1rem', color: epsilon > 25.0 ? 'var(--accent-rose)' : 'var(--text-primary)', marginBottom: '0.4rem' }}>Simulated Model Inversion Attack</h3>
            <p style={{ fontSize: '0.75rem', marginBottom: '0.75rem' }}>What a hacker sees reconstructing images from raw gradients.</p>
            
            <div style={{ 
              width: '100%', flex: 1, minHeight: '160px', 
              background: 'var(--bg-glass)',
              borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)',
              position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <div style={{ position: 'absolute', inset: 0, backgroundImage: 'url(/intercepted_scan.png)', backgroundSize: 'cover', backgroundPosition: 'center' }} />
              <div style={{ position: 'absolute', inset: 0, backdropFilter: `blur(${blurAmount}px)`, backgroundColor: `rgba(5,5,12,${opacityAmount})`, transition: 'all 0.3s', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {epsilon <= 5.0 && <Lock size={36} color="var(--accent-emerald)" />}
              </div>
            </div>

            <div style={{ marginTop: '0.8rem', padding: '0.5rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', fontWeight: 600 }}>
              {epsilon > 25.0 ? <span style={{ color: 'var(--accent-rose)' }}>⚠️ IDENTITY EXPOSED</span> : 
               epsilon <= 5.0 ? <span style={{ color: 'var(--accent-emerald)' }}>🔒 DEFEATED: Pure noise.</span> : 
               <span style={{ color: 'var(--accent-cyan)' }}>Partial Reconstruction (Obscured)</span>}
            </div>
          </div>

          </div>
      </div>
  );
};


// ==========================================
// Slide 3: Quantum Transport (Step-by-step)
// ==========================================
export const QuantumTransportSlide = () => {
  const [hacked, setHacked] = useState(false);
  const [qStep, setQStep] = useState(0);
  const [showKey, setShowKey] = useState(false);

  const sampleKey = "bvMZb7xvFeoJEB6Digw1kLT69OjMT0_SnX5nGooy2c0=";
  const sampleCipher = "gAAAAABlZ6qSe0Z7QwAAAFoxuPM3nITtUHr3pOTdJdC2xV7ySdE68JZTw5nEipZ9uVfsqEX13ZtQx2wa1Hdwg9A3gnYJ2T9m0Q==";
  const plaintext = `{
  "round": 7,
  "dataset": "HAM10000_patchset_B",
  "grads": [0.122, -0.331, 0.044, 0.910, -0.287]
}`;

  const maskedKey = showKey ? sampleKey : "*".repeat(sampleKey.length);
  const cipherLines = sampleCipher.match(/.{1,44}/g) || [sampleCipher];

  const handleNextStep = () => { if (qStep < 2) setQStep(qStep + 1); };
  const resetTarget = () => { setHacked(false); setQStep(0); };

  const stageMeta = [
    { key: 'entangle', title: 'Photon Entanglement',   desc: 'Mint paired Bell states.', icon: <Zap size={16} /> },
    { key: 'chsh',     title: 'CHSH Integrity Test',    desc: 'Detect eavesdroppers.',    icon: <Shield size={16} /> },
    { key: 'fernet',   title: 'Fernet Lock & Uplink',   desc: 'Derive AES key & push.',   icon: <Lock size={16} /> },
  ];

  const getStatus = (idx) => {
    if (hacked) { return idx <= 0 ? (qStep === 0 ? 'active' : 'complete') : idx === 1 ? 'breached' : 'blocked'; }
    if (qStep > idx || (qStep === 2 && idx === 2)) return 'complete';
    if (qStep === idx) return 'active';
    return 'pending';
  };

  const statusStyles = {
    pending:  { label: 'Pending',     color: 'var(--text-dim)',       bg: 'rgba(255,255,255,0.04)' },
    active:   { label: 'In Progress', color: 'var(--accent-cyan)',    bg: 'rgba(0,240,255,0.06)' },
    complete: { label: 'Secured',     color: 'var(--accent-emerald)', bg: 'rgba(16,185,129,0.06)' },
    breached: { label: 'Intercepted', color: 'var(--accent-rose)',    bg: 'rgba(244,63,94,0.08)' },
    blocked:  { label: 'Aborted',     color: 'var(--accent-rose)',    bg: 'rgba(244,63,94,0.05)' },
  };

  const logs = hacked ? [
    "[SYSTEM] Foreign node detected on fiber optic line.", 
    "⚠️ [WARNING] Quantum Wavefunction Collapse!", 
    "⚠️ [CHSH TEST] Score dropped to 1.41 < 2.0!", 
    "🚨 [CRITICAL] EAVESDROPPER DETECTED.", 
    "🚨 [CRITICAL] UPLINK TERMINATED."
  ] : qStep === 0 ? [
    "[SYSTEM] Awaiting Quantum Initialization…"
  ] : qStep === 1 ? [
    "[SYSTEM] E91 Protocol Initiated…", 
    "[SYSTEM] Executing Entanglement…", 
    "[CHSH TEST] Score = 2.82 >> PURE"
  ] : [
    "[SYSTEM] E91 Protocol Initiated…", 
    "✓ [CHSH TEST] Score = 2.82 >> PURE",
    "🔒 [CRYPTO] Wrapping in Fernet (AES-128-CBC)…",
    "🟢 [CRYPTO] Encrypted Payload transmitting."
  ];

  return (
    <div className="animate-slide-up">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Quantum Entanglement Exchange (E91)</h2>
        <p style={{ fontSize: '1rem', maxWidth: '750px' }}>
          Gradients scrambled with DP noise must be transmitted securely. The simulated <strong style={{ color: 'var(--text-primary)' }}>Quantum Fiber-Optic E91 Protocol</strong> guarantees the network cable cannot be tapped.
        </p>
      </div>

      {/* Step Explainers */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        {[
          { title: 'Photon Entanglement', desc: 'Spawn pairs of Quantum Entangled Photons (Bell States). One stays at hospital, its twin travels to the server.', color: 'var(--text-dim)' },
          { title: 'CHSH Security Test', desc: 'Both computers measure their photons. If CHSH Score ≈ 2.82, quantum physics guarantees no hacker. Below 2.0 = hacker detected.', color: 'var(--accent-cyan)' },
          { title: 'Fernet (AES) Lock', desc: 'CHSH passed, so measured photons mathematically generate an unbreakable Fernet AES-128 Token to lock the Medical Gradients.', color: 'var(--accent-emerald)' },
        ].map((s, i) => (
          <div key={i} className="glass-panel" style={{ padding: '1.25rem', borderLeft: `3px solid ${s.color}` }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '0.6rem' }}>Step {i + 1}: {s.title}</h4>
            <p style={{ fontSize: '0.82rem', margin: 0 }}>{s.desc}</p>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        
        {/* Controls */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '0.95rem', marginBottom: '0.5rem' }}>Quantum Controls</h3>
          <button className="btn-primary" onClick={handleNextStep} disabled={qStep >= 2 || hacked} style={{ width: '100%', justifyContent: 'center' }}>
            {qStep === 0 ? '1. Entangle Photons' : qStep === 1 ? '2. Verify & Transmit' : '✓ Uplink Complete'}
          </button>
          <button className="btn-danger" onClick={() => setHacked(true)} disabled={hacked} style={{ width: '100%', justifyContent: 'center' }}>
            ⚠ Simulate Hacker
          </button>
          <button className="btn-ghost" onClick={resetTarget} style={{ width: '100%', justifyContent: 'center' }}>
            Reset
          </button>
        </div>

        {/* Main Visualization */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>

          {/* Stage Monitor */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>
            {stageMeta.map((stage, idx) => {
              const statusKey = getStatus(idx);
              const meta = statusStyles[statusKey];
              return (
                <div key={stage.key} style={{ 
                  padding: '0.75rem', borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${meta.color}`, background: meta.bg,
                  position: 'relative', overflow: 'hidden',
                  transition: 'all 0.3s',
                }}>
                  {statusKey === 'active' && <div className="stage-stripes" />}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                    {stage.icon}
                    <span style={{ fontWeight: 600, fontSize: '0.8rem' }}>{stage.title}</span>
                  </div>
                  <div style={{ 
                    display: 'inline-block', padding: '0.2rem 0.5rem', borderRadius: 'var(--radius-pill)',
                    background: meta.bg, color: meta.color, fontSize: '0.65rem', fontWeight: 700,
                  }}>{meta.label}</div>
                </div>
              );
            })}
          </div>

          {/* Crypto Walkthrough */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-glass)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Server size={14} /> Hospital Plaintext Gradients
              </div>
              <pre className="font-mono" style={{ margin: 0, padding: '0.5rem', background: 'rgba(0,0,0,0.4)', borderRadius: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', border: '1px solid var(--border-glass)' }}>{plaintext}</pre>
            </div>
            <div style={{ 
              padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', 
              border: qStep >= 2 ? '1px solid var(--accent-cyan)' : '1px dashed var(--border-glass)',
              boxShadow: qStep >= 2 ? '0 0 16px rgba(0,240,255,0.1)' : 'none',
              transition: 'all 0.4s'
            }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: qStep >= 2 ? 'var(--accent-cyan)' : 'inherit' }}>
                <Key size={14} /> Live AES-128 Fernet Key
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginBottom: '0.4rem' }}>
                {qStep >= 2 ? 'Generated from entangled photons.' : 'Awaiting CHSH Verification...'}
              </div>
              <div className="font-mono" style={{ padding: '0.5rem', background: 'rgba(0,0,0,0.4)', borderRadius: '6px', fontSize: '0.72rem', color: qStep >= 2 ? 'var(--text-primary)' : 'var(--text-secondary)', border: '1px solid var(--border-glass)', letterSpacing: '0.3px', wordBreak: 'break-all' }}>
                {qStep >= 2 ? maskedKey : '********************************************'}
              </div>
              <button 
                className="btn-ghost" 
                onClick={() => setShowKey(!showKey)} 
                disabled={qStep < 2}
                style={{ marginTop: '0.5rem', padding: '0.35rem 0.75rem', fontSize: '0.75rem', opacity: qStep >= 2 ? 1 : 0.5 }}
              >
                {showKey ? 'Hide' : 'Reveal'} Symmetric Key
              </button>
            </div>
          </div>

          {/* Ciphertext */}
          <div style={{ padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-glass)', marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Lock size={14} color={qStep >= 2 ? "var(--accent-emerald)" : "var(--text-dim)"} /> Encrypted Payload Traversing Network
            </div>
            <div className="font-mono" style={{ fontSize: '0.75rem', color: qStep >= 2 ? 'var(--accent-emerald)' : 'var(--text-dim)', wordBreak: 'break-all', lineHeight: 1.5 }}>
              {qStep >= 2 ? cipherLines.map((line, i) => <div key={i}>{line}</div>) : 'Awaiting payload encryption...'}
            </div>
          </div>

          {/* Terminal Logs */}
          <div className="glass-panel font-mono" style={{ padding: '0.75rem', background: 'rgba(0,0,0,0.5)', color: hacked ? 'var(--accent-rose)' : 'var(--accent-emerald)', fontSize: '0.78rem', minHeight: '100px' }}>
            {logs.map((log, i) => (
              <div key={i} style={{ marginBottom: '0.3rem' }}>{log}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};


// ==========================================
// Slide 4: Krum + Trimmed Mean Aggregation
// ==========================================
export const SecureAggregationSlide = () => {
  const [step, setStep] = useState(0);

  const hospitals = [
    { id: 'A', status: 'Honest',       norm: 1200, grads: [0.12, -0.05, 0.44], color: 'var(--accent-cyan)' },
    { id: 'B', status: 'Honest',       norm: 1350, grads: [0.15, -0.03, 0.41], color: 'var(--accent-emerald)' },
    { id: 'C', status: 'POISONED DATA', norm: 8500, grads: [0.99, -0.99, 2.50], color: 'var(--accent-rose)' },
  ];

  const handleNext = () => setStep(s => (s + 1) % 5);

  return (
    <div className="animate-slide-up">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Krum + Trimmed Mean Aggregation</h2>
        <p style={{ fontSize: '1rem', maxWidth: '750px' }}>
          Encrypted vectors arrive at the server, which runs a <strong style={{ color: 'var(--text-primary)' }}>2-Stage Defense</strong>: first <strong style={{ color: 'var(--text-primary)' }}>Multi-Krum</strong> to filter poisoned networks, then <strong style={{ color: 'var(--text-primary)' }}>Trimmed Mean</strong> to robustly aggregate the survivors.
        </p>
      </div>

      {/* Hospital Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        {hospitals.map((h) => (
          <div key={h.id} className="glass-panel" style={{ 
            padding: '1.25rem', textAlign: 'center', position: 'relative', overflow: 'hidden',
            opacity: step >= 2 && h.id === 'C' ? 0.3 : 1,
            border: step >= 2 && h.id === 'C' ? '1px solid rgba(244,63,94,0.3)' : '1px solid var(--border-glass)',
            transition: 'all 0.5s',
          }}>
            {step >= 2 && h.id === 'C' && (
              <div style={{ 
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%) rotate(-15deg)',
                border: '2px solid var(--accent-rose)', color: 'var(--accent-rose)', 
                padding: '0.15rem 0.5rem', fontWeight: 800, fontSize: '1.2rem', zIndex: 10,
                letterSpacing: '0.1em',
              }}>REJECTED</div>
            )}
            <h4 style={{ color: h.color, fontSize: '0.95rem', marginBottom: '0.5rem' }}>Network {h.id}</h4>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: "'JetBrains Mono', monospace" }}>{h.norm}</div>
            <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem', marginBottom: '0.5rem' }}>Norm ||g||</div>
            <div className="font-mono" style={{ fontSize: '0.8rem', color: h.color, padding: '0.3rem', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)' }}>
              [{h.grads.join(', ')}]
            </div>

            {step > 0 && (
              <div style={{ 
                marginTop: '0.75rem', padding: '0.35rem', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem', fontWeight: 600,
                background: h.id === 'C' && step >= 2 ? 'rgba(244,63,94,0.08)' : step === 1 ? 'rgba(255,255,255,0.04)' : 'rgba(16,185,129,0.06)',
                color: h.id === 'C' && step >= 2 ? 'var(--accent-rose)' : step === 1 ? 'var(--text-muted)' : 'var(--accent-emerald)',
                transition: 'all 0.3s',
              }}>
                {step === 1 ? 'Calculating…' : h.id === 'C' ? 'Outlier Quarantined' : '✓ Krum Verified'}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Protocol Control */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.05rem' }}>Aggregation Sequence</h3>
          <button className="btn-primary" onClick={handleNext}>
            {step === 0 ? "1. Calculate Trajectories" : 
             step === 1 ? "2. Trigger Multi-Krum" : 
             step === 2 ? "3. Apply Trimmed Mean" : 
             step === 3 ? "4. Finalize Global Update" : "Reset Defense"}
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          {/* Krum */}
          <div style={{ 
            padding: '1.25rem', borderRadius: 'var(--radius-md)', 
            border: step >= 1 ? '1px solid rgba(0,240,255,0.2)' : '1px dashed var(--border-glass)',
            background: step >= 1 ? 'rgba(0,240,255,0.03)' : 'transparent',
            transition: 'all 0.4s',
          }}>
            <h4 style={{ fontSize: '0.9rem', color: step >= 1 ? 'var(--text-primary)' : 'var(--text-dim)', marginBottom: '0.75rem' }}>Stage 1: Multi-Krum Filtering</h4>
            <ul style={{ paddingLeft: '1rem', fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.7, margin: 0 }}>
              <li style={{ color: step >= 1 ? 'var(--accent-cyan)' : '' }}>Calculate pairwise spatial distances.</li>
              <li style={{ color: step >= 2 ? 'var(--accent-cyan)' : '' }}>Score networks by closest neighbors.</li>
              <li style={{ color: step >= 2 ? 'var(--text-primary)' : '', fontWeight: step >= 2 ? 600 : 400 }}>
                A & B verified. <span style={{ color: step >= 2 ? 'var(--accent-rose)' : '' }}>C permanently purged.</span>
              </li>
            </ul>
          </div>
          {/* Trimmed Mean */}
          <div style={{ 
            padding: '1.25rem', borderRadius: 'var(--radius-md)', 
            border: step >= 3 ? '1px solid rgba(16,185,129,0.2)' : '1px dashed var(--border-glass)',
            background: step >= 3 ? 'rgba(16,185,129,0.03)' : 'transparent',
            transition: 'all 0.4s',
          }}>
            <h4 style={{ fontSize: '0.9rem', color: step >= 3 ? 'var(--text-primary)' : 'var(--text-dim)', marginBottom: '0.75rem' }}>Stage 2: Trimmed Mean Averaging</h4>
            <ul style={{ paddingLeft: '1rem', fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.7, margin: 0 }}>
              <li style={{ color: step >= 3 ? 'var(--accent-emerald)' : '' }}>Analyze remaining A & B vectors.</li>
              <li style={{ color: step >= 3 ? 'var(--accent-emerald)' : '' }}>Discard upper/lower 10% extremes.</li>
              <li style={{ color: step >= 3 ? 'var(--accent-emerald)' : '' }}>Compute mean of remaining params.</li>
            </ul>
            {step >= 3 && (
              <div className="font-mono animate-slide-up" style={{ marginTop: '0.75rem', padding: '0.6rem', background: 'rgba(0,0,0,0.4)', borderRadius: '6px', border: '1px solid rgba(16,185,129,0.15)', fontSize: '0.78rem' }}>
                <div>A: [{hospitals[0].grads.join(', ')}]</div>
                <div style={{ paddingBottom: '0.4rem', borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: '0.4rem' }}>B: [{hospitals[1].grads.join(', ')}]</div>
                <div style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>Avg: [0.135, -0.040, 0.425]</div>
              </div>
            )}
          </div>
        </div>
      </div>
       
      {step === 4 && (
        <div className="animate-slide-up glass-panel" style={{ padding: '1rem', textAlign: 'center', border: '1px solid rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.05)' }}>
          <CheckCircle size={22} style={{ verticalAlign: 'middle', marginRight: '8px', color: 'var(--accent-emerald)' }} />
          <span style={{ color: 'var(--accent-emerald)', fontWeight: 700, fontSize: '1.05rem' }}>Final Layer Aggregated — Server Model Upgraded!</span>
        </div>
      )}
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

  const tooltipStyle = { 
    background: 'rgba(5,5,12,0.95)', 
    border: '1px solid var(--border-glass)',
    borderRadius: '8px',
    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    fontSize: '0.8rem',
  };

  return (
    <div className="animate-slide-up">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Security Outputs & Performance Analysis</h2>
      </div>
      
      {/* Explanation */}
      <div className="glass-panel" style={{ 
        padding: '1.25rem', marginBottom: '2rem',
        borderLeft: '3px solid var(--accent-cyan)',
      }}>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.9rem' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Left Graph (Convergence):</strong> Shows how quickly the AI learns over 10 rounds. The blue line (Quantum E91 Secured) closely tracks the red line (Unsecured), proving our security layers don't break learning ability!
        </p>
        <p style={{ margin: 0, fontSize: '0.9rem' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Right Graph (Cost of Privacy):</strong> We sacrifice ~10% accuracy to guarantee hackers can never identify patients — the literal mathematical "Cost of Privacy."
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="glass-panel" style={{ padding: '1.5rem', height: '340px' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '1.5rem' }}>Convergence Rate</h3>
          <ResponsiveContainer width="100%" height="80%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="round" stroke="var(--text-dim)" tick={{ fontSize: 12 }} />
              <YAxis stroke="var(--text-dim)" domain={[0, 100]} tick={{ fontSize: 12 }} />
              <RechartsTooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="acc" name="Unsecured" stroke="var(--accent-rose)" strokeWidth={2.5} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="secAcc" name="Quantum E91 Secured" stroke="var(--accent-cyan)" strokeWidth={2.5} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem', height: '340px' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '1.5rem' }}>Cost of Privacy Trade-Off</h3>
          <ResponsiveContainer width="100%" height="80%">
            <BarChart data={[{ name: 'Accuracy Ceiling', unsecured: 85.0, secured: 74.8 }]}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" stroke="var(--text-dim)" tick={{ fontSize: 12 }} />
              <YAxis stroke="var(--text-dim)" domain={[0, 100]} tick={{ fontSize: 12 }} />
              <RechartsTooltip contentStyle={tooltipStyle} />
              <Bar dataKey="unsecured" name="Standard Model" fill="var(--accent-rose)" radius={[6, 6, 0, 0]} />
              <Bar dataKey="secured" name="Quantum E91 Model" fill="var(--accent-cyan)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
