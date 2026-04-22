import React, { useState } from 'react';
import { Shield, Cpu, Activity, LayoutDashboard, ChevronRight, ChevronLeft, Layers, Lock, Eye, BarChart3, Workflow } from 'lucide-react';
import { 
  DataDistributionSlide, 
  LocalTrainingSlide, 
  QuantumTransportSlide, 
  SecureAggregationSlide 
} from './components/Slides';
import { FullPipelineJourney } from './components/FullPipelineJourney';
import { QuantumTransportSimulator } from './components/HybridTransportSimulator';

const NAV_ITEMS = [
  { icon: Layers,   label: 'E91 Architecture',   shortLabel: 'Architecture' },
  { icon: Shield,   label: 'Differential Privacy', shortLabel: 'DP Noise' },
  { icon: Lock,     label: 'Quantum Encryption',  shortLabel: 'E91 Encrypt' },
  { icon: Eye,      label: 'Krum + Trimmed Mean', shortLabel: 'Hybrid Defense' },
  { icon: Workflow, label: 'Full Pipeline',       shortLabel: 'Pipeline' },
];

const App = () => {
  const [activeStep, setActiveStep] = useState(0);

  const slides = [
    <DataDistributionSlide />,
    <LocalTrainingSlide />,
    <QuantumTransportSimulator />,
    <SecureAggregationSlide />,
    <FullPipelineJourney />,
  ];

  const handleNext = () => {
    if (activeStep < slides.length - 1) setActiveStep(activeStep + 1);
  };

  const handlePrev = () => {
    if (activeStep > 0) setActiveStep(activeStep - 1);
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', position: 'relative', zIndex: 1 }}>
      
      {/* ============ LEFT SIDEBAR ============ */}
      <aside style={{
        width: '260px',
        minWidth: '260px',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(8, 10, 18, 0.85)',
        backdropFilter: 'blur(30px)',
        borderRight: '1px solid var(--border-glass)',
        padding: '1.5rem 0',
        zIndex: 10,
      }}>
        {/* Brand */}
        <div style={{ padding: '0 1.25rem 1.5rem', borderBottom: '1px solid var(--border-glass)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
            <div style={{
              width: '36px', height: '36px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-violet) 100%)',
              display: 'grid', placeItems: 'center',
              boxShadow: '0 0 16px rgba(0, 240, 255, 0.25)',
            }}>
              <LayoutDashboard size={18} color="#000" strokeWidth={2.5} />
            </div>
            <div>
              <h1 style={{ fontSize: '1.1rem', fontWeight: 800, margin: 0, letterSpacing: '-0.03em' }}>FLQC Defense</h1>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Quantum E91 Protocol</div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, padding: '1rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '4px', overflowY: 'auto' }}>
          <div style={{ fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '0 0.5rem', marginBottom: '0.5rem' }}>
            Security Layers
          </div>
          {NAV_ITEMS.map((item, index) => {
            const Icon = item.icon;
            const isActive = index === activeStep;
            const isCompleted = index < activeStep;
            return (
              <button
                key={index}
                onClick={() => setActiveStep(index)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  width: '100%',
                  padding: '0.65rem 0.75rem',
                  background: isActive ? 'rgba(0, 240, 255, 0.08)' : 'transparent',
                  border: isActive ? '1px solid rgba(0, 240, 255, 0.2)' : '1px solid transparent',
                  borderRadius: '10px',
                  color: isActive ? 'var(--accent-cyan)' : isCompleted ? 'var(--accent-emerald)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.25s',
                  textAlign: 'left',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 600 : 400,
                  position: 'relative',
                }}
              >
                <div style={{
                  width: '30px', height: '30px',
                  borderRadius: '8px',
                  background: isActive ? 'rgba(0, 240, 255, 0.12)' : isCompleted ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255, 255, 255, 0.04)',
                  display: 'grid', placeItems: 'center',
                  transition: 'all 0.25s',
                  flexShrink: 0,
                }}>
                  <Icon size={15} />
                </div>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.label}</span>
                {isActive && (
                  <div style={{
                    position: 'absolute', left: '-0.75rem', top: '50%', transform: 'translateY(-50%)',
                    width: '3px', height: '60%',
                    borderRadius: '0 3px 3px 0',
                    background: 'var(--accent-cyan)',
                    boxShadow: '0 0 8px var(--accent-cyan)',
                  }} />
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer Info */}
        <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid var(--border-glass)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-emerald)', boxShadow: '0 0 6px var(--accent-emerald)' }} />
            HAM10000 · 7 Classes · 3 Hospitals
          </div>
          <div style={{ marginTop: '0.5rem', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
            Qiskit CHSH · Fernet AES · Gaussian DP
          </div>
        </div>
      </aside>

      {/* ============ MAIN CONTENT ============ */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        
        {/* Top Bar */}
        <header style={{
          padding: '1rem 2rem',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          borderBottom: '1px solid var(--border-glass)',
          background: 'rgba(8, 10, 18, 0.5)',
          backdropFilter: 'blur(16px)',
          zIndex: 5,
          minHeight: '60px',
        }}>
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0 }}>
              {NAV_ITEMS[activeStep].label}
            </h2>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Layer {activeStep} of {slides.length - 1}
            </div>
          </div>

          {/* Progress bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ width: '180px', height: '4px', borderRadius: '2px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${((activeStep + 1) / slides.length) * 100}%`,
                background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-violet))',
                borderRadius: '2px',
                transition: 'width 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                boxShadow: '0 0 8px rgba(0, 240, 255, 0.3)',
              }} />
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>
              {activeStep + 1}/{slides.length}
            </span>
          </div>
        </header>

        {/* Scrollable Slide Area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '2rem 2.5rem 4rem' }}>
          <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
            <div key={activeStep} className="animate-slide-up">
              {slides[activeStep]}
            </div>
          </div>
        </div>

        {/* Bottom Navigation */}
        <footer style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '1rem 2rem',
          borderTop: '1px solid var(--border-glass)',
          background: 'rgba(8, 10, 18, 0.6)',
          backdropFilter: 'blur(16px)',
        }}>
          <button 
            className="btn-ghost"
            onClick={handlePrev} 
            disabled={activeStep === 0}
          >
            <ChevronLeft size={18} /> Previous
          </button>

          <div style={{ display: 'flex', gap: '6px' }}>
            {NAV_ITEMS.map((_, i) => (
              <div
                key={i}
                onClick={() => setActiveStep(i)}
                style={{
                  width: i === activeStep ? '20px' : '8px',
                  height: '8px',
                  borderRadius: '4px',
                  background: i === activeStep ? 'var(--accent-cyan)' : i < activeStep ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.1)',
                  cursor: 'pointer',
                  transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                  boxShadow: i === activeStep ? '0 0 8px var(--accent-cyan)' : 'none',
                }}
              />
            ))}
          </div>

          <button 
            className="btn-primary"
            onClick={handleNext}
            disabled={activeStep === slides.length - 1}
          >
            Next Phase <ChevronRight size={18} />
          </button>
        </footer>
      </main>
    </div>
  );
};

export default App;
