import React, { useState } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import UIOverlay from './components/UIOverlay';
import { SimulationParams, DEFAULT_PARAMS } from './types';

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);

  return (
    <div className="w-full h-screen bg-black overflow-hidden relative font-sans">
      <SimulationCanvas params={params} />
      <UIOverlay params={params} setParams={setParams} />
      
      {/* Title Overlay */}
      <div className="absolute top-6 left-6 pointer-events-none max-w-lg hidden md:block">
        <h1 className="text-3xl font-bold text-white tracking-tight drop-shadow-md">
          Experiment: <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Free Energy Morphology</span>
        </h1>
        <div className="mt-2 bg-black/40 backdrop-blur-sm p-4 rounded-lg border-l-2 border-cyan-500">
           <p className="text-gray-300 text-sm leading-relaxed">
            "We examine whether a dynamic system of particles—characterized by spatial position, activation state, intrinsic spin, and vibrational phase—can facilitate self-supervised learning through free energy minimization."
           </p>
           <p className="text-gray-500 text-xs mt-2 font-mono">
             — Rawson, K. (2025). L-Group Predictive Coding Networks...
           </p>
        </div>
      </div>
    </div>
  );
}

export default App;