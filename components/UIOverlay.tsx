import React, { useState } from 'react';
import { SimulationParams, DEFAULT_PARAMS } from '../types';

interface UIOverlayProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams }) => {
  const [localText, setLocalText] = useState("");
  const [showInfo, setShowInfo] = useState(false);

  const handleChange = (key: keyof SimulationParams, value: number | string | object) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const reset = () => {
    setParams(DEFAULT_PARAMS);
    setLocalText("");
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleChange('inputText', localText);
  };
  
  const clearInput = () => {
    setLocalText("");
    handleChange('inputText', "");
  };

  const forgetCurrentMemory = () => {
    handleChange('memoryResetTrigger', params.memoryResetTrigger + 1);
  };

  // Adaptive Control Logic for Plasticity
  const togglePlasticity = (active: boolean) => {
      if (active) {
          // Adaptive: Set optimal parameters for learning
          setParams(prev => ({
              ...prev,
              plasticity: 0.1,
              damping: 0.95, // High damping to prevent chaotic oscillation during formation
              dataGravity: Math.max(prev.dataGravity, 0.5) // Ensure enough gravity to pull shape
          }));
      } else {
          // Adaptive: Restore parameters for recall/stability
          setParams(prev => ({
              ...prev,
              plasticity: 0,
              damping: 0.85, // Lower damping for natural movement
          }));
      }
  };

  const handleMemoryAction = (type: 'save' | 'load', slot: number) => {
    setParams(prev => ({
      ...prev,
      memoryAction: { type, slot, triggerId: prev.memoryAction.triggerId + 1 }
    }));
  };

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end">
      <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded-xl p-5 text-white w-full shadow-2xl pointer-events-auto overflow-y-auto max-h-[90vh] custom-scrollbar">
        <h1 className="text-xl font-bold mb-1 text-cyan-400">Holographic Memory</h1>
        <p className="text-[10px] text-gray-400 mb-4 italic">Associative Storage in Particle Clouds</p>
        
        {/* 1. Data Input */}
        <div className="mb-4 bg-cyan-900/20 p-3 rounded-lg border border-cyan-500/30">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-cyan-300 mb-2">1. Encode Data</h2>
          <form onSubmit={handleTextSubmit} className="flex gap-2 mb-2">
            <input 
              type="text" 
              value={localText}
              onChange={(e) => setLocalText(e.target.value)}
              placeholder="e.g. QUBIT"
              className="w-full bg-black/50 border border-cyan-500/50 rounded px-2 py-1 text-sm focus:outline-none focus:border-cyan-400 font-mono"
              maxLength={8}
            />
            <button type="submit" className="bg-cyan-600 hover:bg-cyan-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors">
              INPUT
            </button>
          </form>
          
          {/* Data Gravity Slider moved here for context */}
          <div className="mb-2 pt-2 border-t border-cyan-500/20">
            <div className="flex justify-between text-[10px] mb-1">
              <span className="font-mono text-cyan-200">Data Gravity</span>
              <span className="text-cyan-300">{params.dataGravity.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={params.dataGravity}
              onChange={(e) => handleChange('dataGravity', parseFloat(e.target.value))}
              className="w-full h-1 bg-cyan-900/50 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>

          {params.inputText && (
               <button onClick={clearInput} className="w-full bg-red-500/20 hover:bg-red-500/40 text-red-200 text-[10px] py-1 rounded transition-colors">
                 Remove Input (Enter Recall Mode)
               </button>
          )}
        </div>

        {/* 2. Plasticity / Learning */}
        <div className="mb-4 bg-yellow-900/20 p-3 rounded-lg border border-yellow-500/30">
            <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                    <span className="text-yellow-300 text-[10px] font-bold uppercase tracking-wider">2. Imprint (Learn)</span>
                    <button 
                        onClick={() => setShowInfo(!showInfo)}
                        className="text-yellow-500 hover:text-yellow-300 transition-colors"
                    >
                        <span className="text-xs border border-yellow-500 rounded-full w-4 h-4 flex items-center justify-center">?</span>
                    </button>
                </div>
                <div className={`w-2 h-2 rounded-full ${params.plasticity > 0 ? 'bg-yellow-400 animate-pulse' : 'bg-gray-600'}`}></div>
            </div>
            
            {showInfo && (
                <div className="mb-3 p-2 bg-black/40 rounded border border-yellow-500/20 text-[10px] text-gray-300 leading-relaxed shadow-inner">
                    <p className="mb-1"><strong>Plasticity:</strong> Hardens connections between particles based on their current positions.</p>
                </div>
            )}

            <div className="flex items-center gap-2 mb-2">
               <button 
                  onClick={() => togglePlasticity(true)} 
                  className={`flex-1 py-1 rounded text-xs font-bold transition-colors ${params.plasticity > 0 ? 'bg-yellow-500 text-black' : 'bg-gray-700 text-gray-300'}`}
               >
                  ON
               </button>
               <button 
                  onClick={() => togglePlasticity(false)} 
                  className={`flex-1 py-1 rounded text-xs font-bold transition-colors ${params.plasticity === 0 ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-500'}`}
               >
                  OFF
               </button>
            </div>
            <p className="text-[9px] text-yellow-500/60 text-center italic">
                {params.plasticity > 0 ? "Optimizing parameters for learning..." : "Standard physics active"}
            </p>
        </div>

        {/* 3. Memory Bank (New) */}
        <div className="mb-4 bg-purple-900/20 p-3 rounded-lg border border-purple-500/30">
             <h2 className="text-[10px] font-bold uppercase tracking-widest text-purple-300 mb-3">3. Associative Memory Bank</h2>
             
             <div className="space-y-2">
                {/* Slot 1 */}
                <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-purple-200 font-mono">Slot 1</span>
                    <div className="flex gap-1">
                        <button 
                            onClick={() => handleMemoryAction('save', 1)}
                            className="bg-purple-700/50 hover:bg-purple-600 text-[10px] px-2 py-1 rounded text-purple-100 border border-purple-500/50"
                        >
                            Save
                        </button>
                         <button 
                            onClick={() => handleMemoryAction('load', 1)}
                            className="bg-purple-500 hover:bg-purple-400 text-[10px] px-3 py-1 rounded text-white font-bold"
                        >
                            Recall
                        </button>
                    </div>
                </div>

                {/* Slot 2 */}
                <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-purple-200 font-mono">Slot 2</span>
                    <div className="flex gap-1">
                        <button 
                            onClick={() => handleMemoryAction('save', 2)}
                            className="bg-purple-700/50 hover:bg-purple-600 text-[10px] px-2 py-1 rounded text-purple-100 border border-purple-500/50"
                        >
                            Save
                        </button>
                         <button 
                            onClick={() => handleMemoryAction('load', 2)}
                            className="bg-purple-500 hover:bg-purple-400 text-[10px] px-3 py-1 rounded text-white font-bold"
                        >
                            Recall
                        </button>
                    </div>
                </div>
             </div>
             
             <button 
               onClick={forgetCurrentMemory} 
               className="w-full mt-3 py-1 bg-white/5 hover:bg-white/10 text-gray-400 border border-white/10 rounded transition-colors text-[10px] uppercase tracking-wider"
            >
              Clear Current State
            </button>
        </div>

        <div className="h-px bg-gray-700 mb-4" />

        {/* Physics Controls */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-[10px] mb-1">
              <span className="font-mono text-gray-300">Damping (Friction)</span>
              <span className="text-purple-300">{params.damping.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.99"
              step="0.01"
              value={params.damping}
              onChange={(e) => handleChange('damping', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          <div>
            <div className="flex justify-between text-[10px] mb-1">
              <span className="font-mono text-gray-300">Equilibrium Dist (r₀)</span>
              <span className="text-green-300">{params.equilibriumDistance.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.1"
              value={params.equilibriumDistance}
              onChange={(e) => handleChange('equilibriumDistance', parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
        
          <button 
            onClick={reset}
            className="w-full py-2 bg-red-900/20 hover:bg-red-900/40 text-red-300 border border-red-500/20 rounded transition-colors text-[10px] uppercase tracking-wider"
          >
            Hard Reset
          </button>
        </div>
      </div>
    </div>
  );
};

export default UIOverlay;