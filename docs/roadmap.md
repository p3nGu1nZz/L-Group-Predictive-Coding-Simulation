# Roadmap: From Holographic Storage to Quantum Predictive Coding AI

**Status:** Technical Specification 7.0  
**Objective:** Evolve the current Attractor Network into a Hetero-Associative Computational Inference Engine.

---

## 1. Implementation Phases (Sprint Plan)

### Sprint 0: Semantic Layering (Architecture)
**Status:** ✅ COMPLETE
*   Partitioned particles into Regions A, B, and Associative.
*   Implemented core attractor dynamics.

### Sprint 1: Agency & Time (The "Living" Agent)
**Status:** ✅ COMPLETE
*   **Active Inference:** Implemented Thermodynamic Agency (Auto-Agitate/Freeze).
*   **Temporal Dynamics:** Implemented STDP and Asymmetric Weights (`forwardMatrix`).
*   **Deliverables:** Experiment A (Inference) and Experiment B (Prediction).

---

### Sprint 2: Quantum Logic (Computation)
**Status:** 🚧 IN PROGRESS
**Goal:** Implement non-linear logic gates (XOR) using hysteresis and interference.

**Implementation Guide:**
1.  **Hysteresis (Latching):**
    *   Implement a "Schmitt Trigger" for particle activation.
    *   Once a particle turns ON, it requires significantly less input to *stay* ON than it did to turn ON initially.
    *   This allows the system to hold a state (Memory Latching).
2.  **Logic Gates (Experiment C):**
    *   **AND Gate:** Output Region only activates if Input A AND Input B provide energy.
    *   **XOR Gate:** Output Region activates for A or B, but *deactivates* via inhibition if BOTH are active.
3.  **Complex Amplitude:**
    *   Visualize the "Magnitude" of the state vector to show destructive vs constructive interference.

**Acceptance Criteria:**
*   A Truth Table visualization that updates in real-time.
*   Experiment C successfully demonstrates an XOR operation.

---

### Sprint 3: Optimization & Scaling
**Status:** 📅 PLANNED
**Goal:** Scale to N > 2000 using spatial hashing optimizations.

**Implementation Guide:**
1.  **Spatial Index:** 
    *   Refine Grid Cell algorithm for massive particle counts.
    *   Offload physics to WebWorker.
2.  **Holographic Capacity:** 
    *   Stress test the network to see how many unique logic gates can be superimposed before "forgetting" occurs.

---

## 2. Technical Architecture & Data Structures

### Extended Data Model
**File:** `types.ts`
```typescript
export interface ParticleData {
  // ... existing fields
  hysteresisState: Uint8Array;     // 0 = Off, 1 = On (Latching)
  complexAmplitude: Float32Array;  // For visualization of interference
}
```

### Constants & Parameter Defaults
```typescript
export const CONSTANTS = {
  // Logic
  ampThresholdOn: 0.8,    // Turn ON threshold
  ampThresholdOff: 0.4,   // Turn OFF threshold (Hysteresis gap)
};
```

---

## 3. Testing Strategy

### Automated Logic Tests
*   **XOR Test:**
    1. Input A=0, B=0 -> Output=0
    2. Input A=1, B=0 -> Output=1
    3. Input A=0, B=1 -> Output=1
    4. Input A=1, B=1 -> Output=0 (Critical Failure Point for linear networks)
