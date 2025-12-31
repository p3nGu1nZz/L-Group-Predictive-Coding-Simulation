export type MemoryActionType = 'idle' | 'save' | 'load' | 'clear';

export interface MemoryAction {
  type: MemoryActionType;
  slot: number;
  triggerId: number; 
}

export interface SimulationParams {
  particleCount: number;
  equilibriumDistance: number; 
  stiffness: number; 
  couplingDecay: number; 
  phaseSyncRate: number; 
  spatialLearningRate: number; 
  dataGravity: number; 
  plasticity: number; 
  damping: number; 
  inputText: string; 
  targetRegion: number; // -1 = All, 0 = Region A, 1 = Region B
  memoryResetTrigger: number; 
  memoryAction: MemoryAction; 
  paused: boolean; // New Flag
}

export interface ParticleData {
  x: Float32Array; 
  v: Float32Array; 
  phase: Float32Array; 
  spin: Float32Array; 
  activation: Float32Array; 
  target: Float32Array; 
  hasTarget: Uint8Array;
  memoryMatrix: Float32Array; 
  regionID: Uint8Array;
  forwardMatrix: Float32Array;
  feedbackMatrix: Float32Array;
  delayedActivation: Float32Array;
  lastActiveTime: Float32Array;
}

export const CONSTANTS = {
  // Physics
  couplingDecay: 4.0,
  baseStiffness: 0.8,
  
  // Learning
  basePlasticity: 0.02,
  noiseScale: 0.1,
  consolidationThreshold: 0.05, // Temp < 0.05 triggers save
  
  // Temporal
  delayAlpha: 0.1,        // Activation smoothing
  stdpWindow: 0.050,      // 50ms (in seconds)
  
  // Logic
  ampThresholdOn: 0.8,    // Hysteresis ON
  ampThresholdOff: 0.4,   // Hysteresis OFF
  
  // Optimization
  gridCellSize: 6.0,      // ~1.5x couplingDecay
  spatialRefreshRate: 10, // Frames
};

export const DEFAULT_PARAMS: SimulationParams = {
  particleCount: 800, 
  equilibriumDistance: 1.0, 
  stiffness: CONSTANTS.baseStiffness,
  couplingDecay: CONSTANTS.couplingDecay,
  phaseSyncRate: 0.05,
  spatialLearningRate: 0.05,
  dataGravity: 0.2, 
  plasticity: 0.0, 
  damping: 0.80, 
  inputText: "",
  targetRegion: -1, // Default to Global
  memoryResetTrigger: 0,
  memoryAction: { type: 'idle', slot: 0, triggerId: 0 },
  paused: false,
};