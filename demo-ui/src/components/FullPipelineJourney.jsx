import React, { useState } from 'react';
import { Database, Shield, Lock, Server, Cpu, Activity, ArrowRight, CheckCircle, ChevronRight } from 'lucide-react';

const STAGES = [
  { label: 'Generate Gradient',  icon: Database, color: 'var(--text-primary)',   blob: '... Awaiting Local Hospital Data ...', blobClass: '' },
  { label: 'Local AI Training',  icon: Database, color: 'var(--text-primary)',   blob: 'RAW: [0.45, 0.12, 0.89]', blobClass: 'raw' },
  { label: 'Inject DP Noise',    icon: Shield,   color: 'var(--accent-purple)',  blob: 'NOISE: [-2.12, +4.88, -0.05]', blobClass: 'dp' },
  { label: 'E91 Quantum Lock',   icon: Lock,     color: 'var(--accent-cyan)',    blob: 'E91: gAAAAABkV…', blobClass: 'encrypt' },
  { label: 'Transmit Payload',   icon: Activity,  color: 'var(--accent-cyan)',   blob: 'Transmitting over Internet…', blobClass: 'encrypt transmit' },
  { label: 'Krum Verification',  icon: Server,   color: 'var(--accent-emerald)', blob: 'TRUSTED KRUM DISTANCE', blobClass: 'verified' },
  { label: 'Merge to Global',    icon: Cpu,      color: 'gold',                  blob: 'MERGED_TO_GLOBAL_MODEL', blobClass: 'global' },
];

const PIPELINE_NODES = [
  { label: '1. Local Hospital',   desc: 'Train local AI on raw patient scans.', icon: Database, activeAt: 1 },
  { label: '2. Privacy Engine',   desc: 'Inject Gaussian Noise (ε=5.0).',       icon: Shield,   activeAt: 2 },
  { label: '3. Quantum E91 Lock', desc: 'Verify CHSH and wrap in AES-256.',     icon: Lock,     activeAt: 3 },
  { label: '4. Krum Aggregation', desc: 'Compute Euclidean distances.',          icon: Server,   activeAt: 5 },
  { label: '5. Global Model',     desc: 'Safely update global intelligence.',   icon: Cpu,      activeAt: 6 },
];

export const FullPipelineJourney = () => {
  const [stage, setStage] = useState(0);

  const nextStage = () => { if (stage < 6) setStage(stage + 1); };
  const resetStage = () => setStage(0);

  const currentStage = STAGES[stage];
  const CurrentBlobIcon = currentStage.icon;

  return (
    <div className="animate-slide-up">
      <style>{`
        .data-blob {
          padding: 0.6rem 1.25rem; border-radius: var(--radius-md); font-weight: 700;
          font-family: 'JetBrains Mono', monospace; display: flex; align-items: center;
          gap: 0.6rem; font-size: 0.95rem; transition: all 0.5s ease;
          background: rgba(255,255,255,0.04); border: 1px solid var(--border-glass); color: var(--text-muted);
        }
        .data-blob.raw      { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.15); color: var(--text-primary); }
        .data-blob.dp        { background: rgba(168,85,247,0.08); border-color: rgba(168,85,247,0.3); color: var(--accent-purple); box-shadow: 0 0 16px rgba(168,85,247,0.15); }
        .data-blob.encrypt   { background: rgba(0,240,255,0.08); border-color: rgba(0,240,255,0.3); color: var(--accent-cyan); box-shadow: 0 0 16px rgba(0,240,255,0.15); }
        .data-blob.transmit  { animation: dataPulse 1.5s infinite; }
        .data-blob.verified  { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.3); color: var(--accent-emerald); box-shadow: 0 0 16px rgba(16,185,129,0.15); }
        .data-blob.global    { background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.3); color: var(--accent-gold); box-shadow: 0 0 20px rgba(251,191,36,0.15); transform: scale(1.05); }
        @keyframes dataPulse { 0%,100% { opacity: 0.6; transform: scale(0.98); } 50% { opacity: 1; transform: scale(1.02); } }
      `}</style>
      
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="title-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Full Lifecycle Security Simulation</h2>
        <p style={{ fontSize: '1rem', maxWidth: '700px' }}>
          Step-by-step trace of a patient gradient matrix from Hospital creation to Global Aggregation.
        </p>
      </div>

      {/* Data Packet Display */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-dim)', fontWeight: 600 }}>
          Live Data Packet Memory Buffer
        </div>
        <div className={`data-blob ${currentStage.blobClass}`}>
          <CurrentBlobIcon size={16} />
          {currentStage.blob}
        </div>
      </div>

      {/* Pipeline Nodes */}
      <div style={{ display: 'flex', width: '100%', alignItems: 'stretch', gap: '0', marginBottom: '2rem' }}>
        {PIPELINE_NODES.map((node, idx) => {
          const Icon = node.icon;
          const isActive = stage === node.activeAt;
          const isCompleted = stage > node.activeAt;
          const isGlobal = idx === PIPELINE_NODES.length - 1 && stage === 6;
          
          return (
            <React.Fragment key={idx}>
              <div className="glass-panel" style={{
                flex: 1,
                padding: '1.25rem 0.75rem',
                textAlign: 'center',
                border: isActive ? '1px solid var(--accent-cyan)' : isGlobal ? '1px solid rgba(251,191,36,0.4)' : '1px solid var(--border-glass)',
                boxShadow: isActive ? '0 0 20px rgba(0,240,255,0.1)' : isGlobal ? '0 0 20px rgba(251,191,36,0.1)' : 'none',
                opacity: isCompleted && !isGlobal ? 0.5 : 1,
                transform: isActive ? 'translateY(-4px)' : 'none',
                transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
              }}>
                <Icon size={26} color={isGlobal ? 'gold' : isActive ? 'var(--accent-cyan)' : isCompleted ? 'var(--accent-emerald)' : 'var(--text-dim)'} style={{ marginBottom: '0.5rem' }} />
                <h4 style={{ fontSize: '0.8rem', marginBottom: '0.3rem', color: isGlobal ? 'gold' : 'var(--text-primary)' }}>{node.label}</h4>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', margin: 0 }}>{node.desc}</p>
              </div>
              
              {idx < PIPELINE_NODES.length - 1 && (
                <div style={{ 
                  display: 'flex', alignItems: 'center', padding: '0 0.25rem',
                  color: stage > node.activeAt ? 'var(--accent-cyan)' : 'var(--text-dim)',
                  transition: 'color 0.3s',
                }}>
                  <ChevronRight size={18} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
        <button 
          className="btn-primary"
          onClick={nextStage}
          disabled={stage >= 6}
          style={{ padding: '0.8rem 2rem' }}
        >
          {stage === 0 ? 'Start Trace: Generate Gradient' : 
           stage === 1 ? 'Step 1: Inject Differential Privacy' : 
           stage === 2 ? 'Step 2: Encrypt via Quantum E91' : 
           stage === 3 ? 'Step 3: Transmit Payload' : 
           stage === 4 ? 'Step 4: Run Krum Filtering' : 
           stage === 5 ? 'Step 5: Merge into Global' : 'Simulation Complete'}
          {stage < 6 && <ArrowRight size={16} />}
        </button>

        {stage >= 1 && (
          <button className="btn-ghost" onClick={resetStage} style={{ color: 'var(--accent-rose)', borderColor: 'rgba(244,63,94,0.3)' }}>
            Reset Trace
          </button>
        )}
      </div>
    </div>
  );
};
