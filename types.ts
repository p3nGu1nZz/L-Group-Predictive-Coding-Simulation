

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
  paused: boolean;
  showRegions: boolean; 
  chaosMode: boolean; // New flag for entropy injection
  
  // Paper Specifics
  usePaperPhysics: boolean;
  spinCouplingStrength: number; // Gamma in Eq 18
  phaseCouplingStrength: number; // Alpha in Eq 14
}

export interface ParticleData {
  x: Float32Array; 
  v: Float32Array; 
  phase: Float32Array; 
  spin: Int8Array; // Changed to Int8 for -1, 1
  activation: Float32Array; 
  target: Float32Array; 
  hasTarget: Uint8Array;
  memoryMatrix: Float32Array; 
  regionID: Uint8Array;
  forwardMatrix: Float32Array;
  feedbackMatrix: Float32Array;
  delayedActivation: Float32Array;
  lastActiveTime: Float32Array;
  hysteresisState: Uint8Array; // 0 = Off, 1 = On (Latching)
}

export interface MemorySnapshot {
  x: Float32Array;
  regionID: Uint8Array;
  forwardMatrix?: Float32Array; 
}

export interface SystemStats {
  meanError: number;
  meanSpeed: number;
  energy: number;
  fps: number;
  temperature: number;
  isStable: boolean;
  trainingProgress: number; 
  // Advanced Telemetry
  phaseOrder: number;   // Kuramoto Index (0-1): Global Phase Synchronization
  spinOrder: number;    // Magnetization (0-1): Net Spin Alignment
  entropy: number;      // Thermodynamic Entropy proxy
  patternMatch: number; // % Similarity to target state
}

export interface TestResult {
  testName: string;
  score: number;
  maxScore: number;
  status: 'PASS' | 'FAIL';
  details: string;
}

export const CONSTANTS = {
  // Physics
  couplingDecay: 4.0,
  baseStiffness: 0.8,
  
  // Learning
  basePlasticity: 0.02,
  noiseScale: 0.1,
  
  // Optimization
  gridCellSize: 6.0,      // ~1.5x couplingDecay
  spatialRefreshRate: 10, // Frames

  // Logic / Hysteresis
  activationThresholdHigh: 0.65, // Energy needed to turn ON
  activationThresholdLow: 0.3,   // Energy needed to stay ON
};

export const DEFAULT_PARAMS: SimulationParams = {
  particleCount: 1200, 
  equilibriumDistance: 1.0, 
  stiffness: CONSTANTS.baseStiffness,
  couplingDecay: CONSTANTS.couplingDecay,
  phaseSyncRate: 0.05,
  spatialLearningRate: 0.05,
  dataGravity: 0.15, 
  plasticity: 0.0, 
  damping: 0.85, 
  inputText: "QUANTUM",
  targetRegion: -1, // Default to Global
  memoryResetTrigger: 0,
  memoryAction: { type: 'idle', slot: 0, triggerId: 0 },
  paused: false,
  showRegions: true, // Default to true
  chaosMode: false,
  
  usePaperPhysics: false,
  spinCouplingStrength: 0.5,
  phaseCouplingStrength: 1.0,
};