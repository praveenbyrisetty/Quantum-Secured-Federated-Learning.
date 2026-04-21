import React, { useState } from 'react';
import { Network, Shield, Cpu, Activity, LayoutDashboard, ChevronRight, ChevronLeft } from 'lucide-react';
import { 
  DataDistributionSlide, 
  LocalTrainingSlide, 
  QuantumTransportSlide, 
  SecureAggregationSlide, 
  FinalEvaluationSlide 
} from './components/Slides';
import { FullPipelineJourney } from './components/FullPipelineJourney';
import { QuantumTransportSimulator } from './components/QuantumTransportSimulator';

const App = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    { title: 'Layer 0: Core Quantum E91 Architecture', component: <DataDistributionSlide /> },
    { title: 'Layer 1: Local Differential Privacy', component: <LocalTrainingSlide /> },
    { title: 'Layer 2: Quantum E91 Encryption', component: <QuantumTransportSimulator /> },
    { title: 'Layer 3: Krum Defensive Aggregation', component: <SecureAggregationSlide /> },
    { title: 'Layer 4: Final Security Evaluation', component: <FinalEvaluationSlide /> },
    { title: 'Layer 5: Full Lifecycle Security Simulation', component: <FullPipelineJourney /> }
  ];

  const handleNext = () => {
    if (activeStep < steps.length - 1) setActiveStep(activeStep + 1);
  };

  const handlePrev = () => {
    if (activeStep > 0) setActiveStep(activeStep - 1);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden', background: 'radial-gradient(circle at top, rgba(138,43,226,0.1), transparent 60%)' }}>
      
      {/* Top Header */}
      <div className="glass-panel" style={{ 
        padding: '1rem 2rem', 
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderBottom: '1px solid var(--border-glass)', borderRadius: 0, zIndex: 100
      }}>
        <h1 className="title-gradient" style={{ fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
          <LayoutDashboard size={24} color="var(--accent-cyan)" />
          Quantum E91 FLQC Defense
        </h1>
        
        {/* Progress Dots */}
        <div style={{ display: 'flex', gap: '1rem' }}>
          {steps.map((_, index) => (
            <div key={index} style={{
              width: '12px', height: '12px', borderRadius: '50%',
              background: index === activeStep ? 'var(--accent-cyan)' : index < activeStep ? 'var(--accent-emerald)' : 'rgba(255,255,255,0.2)',
              boxShadow: index === activeStep ? '0 0 10px var(--accent-cyan)' : 'none',
              transition: 'all 0.3s'
            }} />
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', justifyContent: 'center', alignItems: 'flex-start', padding: '2rem' }}>
        <div style={{ maxWidth: '1200px', width: '100%' }}>
          {/* Component Render */}
          <div style={{ minHeight: '60vh' }}>
            {steps[activeStep].component}
          </div>

          {/* Step-by-Step Footer Navigation */}
          <div style={{ 
            marginTop: '3rem', paddingTop: '2rem', borderTop: '1px solid var(--border-glass)',
            display: 'flex', justifyContent: 'space-between'
          }}>
            <button 
              onClick={handlePrev} 
              disabled={activeStep === 0}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                padding: '1rem 2rem', background: 'rgba(255,255,255,0.05)', 
                color: activeStep === 0 ? 'rgba(255,255,255,0.2)' : 'var(--text-primary)', 
                border: '1px solid var(--border-glass)', borderRadius: '8px', 
                fontSize: '1.1rem', cursor: activeStep === 0 ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s'
              }}
            >
              <ChevronLeft size={20} /> Previous Phase
            </button>

            <button 
              onClick={handleNext} 
             disabled={activeStep === steps.length - 1}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                padding: '1rem 2rem', background: activeStep === steps.length - 1 ? 'rgba(255,255,255,0.05)' : 'var(--accent-cyan)', 
                color: activeStep === steps.length - 1 ? 'rgba(255,255,255,0.2)' : '#000', 
                border: 'none', borderRadius: '8px', 
                fontSize: '1.1rem', fontWeight: 'bold', 
                cursor: activeStep === steps.length - 1 ? 'not-allowed' : 'pointer',
                boxShadow: activeStep === steps.length - 1 ? 'none' : '0 0 20px rgba(0,240,255,0.4)',
                transition: 'all 0.3s'
              }}
            >
              Next Phase <ChevronRight size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
