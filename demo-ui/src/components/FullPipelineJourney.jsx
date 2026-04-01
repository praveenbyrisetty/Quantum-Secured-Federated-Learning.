import React, { useState } from 'react';
import { Database, Shield, Lock, Server, Cpu, Activity, ArrowRight, CheckCircle } from 'lucide-react';

export const FullPipelineJourney = () => {
  const [stage, setStage] = useState(0);

  const nextStage = () => {
    if (stage < 6) setStage(stage + 1);
  };

  const resetStage = () => setStage(0);

  const renderDataState = () => {
    switch(stage) {
      case 0: return <div className="data-blob" style={{color: 'var(--text-muted)'}}>... Awaiting Local Hospital Data ...</div>;
      case 1: return <div className="data-blob raw"><Database size={16} /> RAW: [0.45, 0.12, 0.89]</div>;
      case 2: return <div className="data-blob dp"><Shield size={16} /> NOISE: [-2.12, +4.88, -0.05]</div>;
      case 3: return <div className="data-blob encrypt"><Lock size={16} /> E91: gAAAAABkV...</div>;
      case 4: return <div className="data-blob encrypt" style={{animation: 'dataPulse 1.5s infinite'}}><Activity size={16} /> Transmitting over Internet...</div>;
      case 5: return <div className="data-blob verified"><CheckCircle size={16} /> TRUSTED KRUM DISTANCE</div>;
      case 6: return <div className="data-blob global"><Cpu size={16} /> MERGED_TO_GLOBAL_MODEL</div>;
      default: return null;
    }
  };

  return (
    <div className="animate-slide-up" style={{ padding: '0 2rem' }}>
      <style>{`
        .data-blob { padding: 0.8rem 1.5rem; border-radius: 8px; font-weight: bold; font-family: monospace; display: flex; align-items: center; gap: 0.8rem; font-size: 1.1rem; box-shadow: 0 0 20px rgba(0,0,0,0.5); transition: all 0.5s ease; }
        .data-blob.raw { background: rgba(255, 255, 255, 0.1); border: 1px solid var(--text-primary); color: var(--text-primary); }
        .data-blob.dp { background: rgba(138, 43, 226, 0.15); border: 1px solid var(--accent-purple); color: var(--accent-purple); box-shadow: 0 0 15px var(--accent-purple); }
        .data-blob.encrypt { background: rgba(0, 240, 255, 0.15); border: 1px solid var(--accent-cyan); color: var(--accent-cyan); box-shadow: 0 0 15px var(--accent-cyan); }
        .data-blob.verified { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent-emerald); color: var(--accent-emerald); box-shadow: 0 0 15px var(--accent-emerald); }
        .data-blob.global { background: rgba(255, 215, 0, 0.15); border: 1px solid gold; color: gold; box-shadow: 0 0 20px gold; transform: scale(1.1); }
        
        .pipeline-card { background: rgba(10, 10, 15, 0.8); border: 1px solid var(--border-glass); border-radius: 12px; padding: 1.5rem; flex: 1; text-align: center; position: relative; transition: all 0.3s; }
        .pipeline-card.active { border-color: var(--accent-cyan); box-shadow: 0 0 25px rgba(0,240,255,0.2); transform: translateY(-5px); }
        .pipeline-card.completed { border-color: var(--accent-emerald); opacity: 0.6; }
        
        .arrow-connector { display: flex; align-items: center; justify-content: center; padding: 0 0.5rem; color: var(--text-muted); transition: all 0.3s; }
        .arrow-connector.active { color: var(--accent-cyan); animation: dataPulse 1.5s infinite; }
        
        @keyframes dataPulse { 0% { opacity: 0.5; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.2); } 100% { opacity: 0.5; transform: scale(0.9); } }
      `}</style>
      
      <h2 className="title-gradient" style={{ fontSize: '2.5rem', marginBottom: '1rem', textAlign: 'center' }}>Layer 5: Full Lifecycle Security Simulation</h2>
      <p style={{ fontSize: '1.2rem', marginBottom: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Step-by-step mathematical trace of a patient gradient matrix from Hospital creation to Global Aggregation.
      </p>

      {/* Main Orchestration UI */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem', alignItems: 'center' }}>
        
        {/* Dynamic Data Packet Display */}
        <div style={{ background: '#000', padding: '1.5rem 3rem', borderRadius: '12px', border: '1px solid var(--border-glass)', width: '80%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', minHeight: '120px', justifyContent: 'center' }}>
           <div style={{ textTransform: 'uppercase', letterSpacing: '2px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>Live Data Packet Memory Buffer</div>
           {renderDataState()}
        </div>

        {/* Pipeline Nodes */}
        <div style={{ display: 'flex', width: '100%', justifyContent: 'space-between', alignItems: 'stretch' }}>
           
           {/* Client Node */}
           <div className={`pipeline-card ${stage === 1 ? 'active' : stage > 1 ? 'completed' : ''}`}>
             <Database size={32} color={stage >= 1 ? "var(--text-primary)" : "var(--text-muted)"} style={{ marginBottom: '1rem' }} />
             <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>1. Local Hospital</h4>
             <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Train local AI on raw patient scans.</p>
           </div>
           
           <div className={`arrow-connector ${stage === 1 ? 'active' : ''}`}><ArrowRight size={24} /></div>

           {/* DP Node */}
           <div className={`pipeline-card ${stage === 2 ? 'active' : stage > 2 ? 'completed' : ''}`}>
             <Shield size={32} color={stage >= 2 ? "var(--accent-purple)" : "var(--text-muted)"} style={{ marginBottom: '1rem' }} />
             <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>2. Privacy Engine</h4>
             <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Inject Gaussian Noise (ε=5.0).</p>
           </div>

           <div className={`arrow-connector ${stage === 2 ? 'active' : ''}`}><ArrowRight size={24} /></div>

           {/* E91 Node */}
           <div className={`pipeline-card ${stage === 3 ? 'active' : stage > 3 ? 'completed' : ''}`}>
             <Lock size={32} color={stage >= 3 ? "var(--accent-cyan)" : "var(--text-muted)"} style={{ marginBottom: '1rem' }} />
             <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>3. Quantum E91 Lock</h4>
             <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Verify CHSH and wrap in AES-256.</p>
           </div>

           <div className={`arrow-connector ${stage === 3 || stage === 4 ? 'active' : ''}`}><ArrowRight size={24} /></div>

           {/* Server/Krum Node */}
           <div className={`pipeline-card ${stage === 5 ? 'active' : stage > 5 ? 'completed' : ''}`}>
             <Server size={32} color={stage >= 5 ? "var(--accent-emerald)" : "var(--text-muted)"} style={{ marginBottom: '1rem' }} />
             <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>4. Krum Aggregation</h4>
             <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Compute Euclidean distances.</p>
           </div>

           <div className={`arrow-connector ${stage === 5 ? 'active' : ''}`}><ArrowRight size={24} /></div>

           {/* Global Model */}
           <div className={`pipeline-card ${stage === 6 ? 'active' : ''}`} style={{ borderColor: stage === 6 ? 'gold' : '' }}>
             <Cpu size={32} color={stage === 6 ? "gold" : "var(--text-muted)"} style={{ marginBottom: '1rem' }} />
             <h4 style={{ marginBottom: '0.5rem', color: stage === 6 ? 'gold' : 'var(--text-primary)' }}>5. Global Model</h4>
             <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Safely update global intelligence.</p>
           </div>

        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
          <button 
             onClick={nextStage}
             disabled={stage >= 6}
             style={{
               padding: '1rem 3rem', background: stage >= 6 ? 'var(--bg-card)' : 'var(--accent-cyan)', 
               color: stage >= 6 ? 'var(--text-muted)' : '#000', 
               border: 'none', borderRadius: '8px', fontSize: '1.2rem', fontWeight: 'bold', 
               cursor: stage >= 6 ? 'not-allowed' : 'pointer', boxShadow: stage < 6 ? '0 0 20px rgba(0,240,255,0.4)' : 'none',
               display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.3s'
             }}
          >
             {stage === 0 ? 'Start Trace: Generate Local AI Gradient' : 
              stage === 1 ? 'Step 1: Inject Differential Privacy' : 
              stage === 2 ? 'Step 2: Encrypt via Quantum E91' : 
              stage === 3 ? 'Step 3: Transmit Payload to Server' : 
              stage === 4 ? 'Step 4: Run Krum Aggregation Filtering' : 
              stage === 5 ? 'Step 5: Merge into Global Baseline' : 'Simulation Complete'}
             {stage < 6 && <ArrowRight size={20} />}
          </button>

          {stage >= 1 && (
            <button 
              onClick={resetStage}
              style={{
                padding: '1rem 2rem', background: 'transparent', color: 'var(--accent-rose)',
                border: '1px solid var(--accent-rose)', borderRadius: '8px', cursor: 'pointer',
                fontSize: '1.1rem', fontWeight: 'bold'
              }}
            >
              Reset Memory Trace
            </button>
          )}
        </div>

      </div>
    </div>
  );
};
