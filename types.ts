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
  memoryResetTrigger: number; 
  memoryAction: MemoryAction; 
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
}

export const DEFAULT_PARAMS: SimulationParams = {
  particleCount: 800, 
  equilibriumDistance: 1.0, 
  stiffness: 0.8,
  couplingDecay: 4.0,
  phaseSyncRate: 0.05,
  spatialLearningRate: 0.05,
  dataGravity: 0.2, 
  plasticity: 0.0, 
  damping: 0.80, 
  inputText: "",
  memoryResetTrigger: 0,
  memoryAction: { type: 'idle', slot: 0, triggerId: 0 },
};