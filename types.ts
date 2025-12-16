export type MemoryActionType = 'idle' | 'save' | 'load' | 'clear';

export interface MemoryAction {
  type: MemoryActionType;
  slot: number;
  triggerId: number; // Increment to trigger effect
}

export interface SimulationParams {
  particleCount: number;
  equilibriumDistance: number; // Global base r0 (if no memory)
  stiffness: number; // k in paper
  couplingDecay: number; // sigma in paper
  phaseSyncRate: number; // kappa
  spatialLearningRate: number; // eta_r
  dataGravity: number; // Input strength
  plasticity: number; // How fast connections learn the current shape (0 to 1)
  damping: number; // Friction (0.0 to 1.0)
  inputText: string; 
  memoryResetTrigger: number; // Signal to wipe current memory
  memoryAction: MemoryAction; // Command for Memory Bank
}

export interface ParticleData {
  x: Float32Array; 
  v: Float32Array; 
  phase: Float32Array; 
  spin: Float32Array; 
  activation: Float32Array; 
  target: Float32Array; 
  hasTarget: Uint8Array;
  // We need a memory matrix. Flattened N*N array storing the "learned" distance between i and j.
  // -1 implies no connection memory.
  memoryMatrix: Float32Array; 
}

export const DEFAULT_PARAMS: SimulationParams = {
  particleCount: 800, 
  equilibriumDistance: 1.0, // Reduced for tighter packing
  stiffness: 0.8,
  couplingDecay: 4.0,
  phaseSyncRate: 0.05,
  spatialLearningRate: 0.05,
  dataGravity: 0.2, 
  plasticity: 0.0, // Default off
  damping: 0.80, // High drag to prevent hairball
  inputText: "",
  memoryResetTrigger: 0,
  memoryAction: { type: 'idle', slot: 0, triggerId: 0 },
};