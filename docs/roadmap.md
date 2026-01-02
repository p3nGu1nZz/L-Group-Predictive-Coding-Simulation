# Roadmap: From Holographic Storage to Quantum Predictive Coding AI

**Status:** Technical Specification 6.0  
**Objective:** Evolve the current Attractor Network (Auto-Associative Memory) into a Hetero-Associative Computational Inference Engine.

---

## 1. Executive Summary

The prototype currently functions as a **Holographic Memory Store** (Auto-Associative). To evolve this into **Artificial Intelligence**, we must implement four prioritized capabilities: **Semantic Layering**, **Reward-Modulated Plasticity**, **Temporal Dynamics**, and **Quantum Logic**.

This specification is an **Actionable Implementation Plan** with explicit parameter defaults, data schemas, and safety protocols to ensure robust engineering.

---

## 2. Technical Architecture & Data Structures

### 2.1 Extended Data Model
**File:** `types.ts`
```typescript
export interface ParticleData {
  // ... existing fields (x, v, phase, etc.)
  regionID: Uint8Array;            // 0=Input A (Cyan), 1=Input B (Pink), 2=Assoc (Gold)
  forwardMatrix: Float32Array;     // Directed excitation (Pre -> Post)
  feedbackMatrix: Float32Array;    // Prediction error (Post -> Pre)
  delayedActivation: Float32Array; // Axonal delay buffer
  lastActiveTime: Float32Array;    // Wall-clock timestamp (seconds) for STDP
}
```

### 2.2 System State & Telemetry
Global variables required for the "Teacher" loop, runtime monitoring, and safety breaks.

*   `meanError`: Running average of output deviation.
*   `Temperature (T)`: [0.0 - 1.0], derived from error.
*   `userFeedback`: -1 (Wrong/Hot), 0 (Neutral), 1 (Correct/Cold).
*   `plasticityRate`: Fraction of edges updated per frame.
*   `plasticityHistory`: Ring buffer of last 60 frames of rates (for safety checks).

### 2.3 Constants & Parameter Defaults
Use these values to prevent tuning instability.

```typescript
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
```

---

## 3. Implementation Phases (Sprint Plan)

### Sprint 0: Semantic Layering (Architecture)
**Goal:** Partition particles and enforce inhibition rules.

**Implementation Guide:**
1.  **Init:** Region 0 (0-25%), Region 1 (25-50%), Region 2 (50-100%).
2.  **Gating:** 
    *   Block 0↔1 coupling.
    *   Harden 0↔0 and 1↔1 bonds (`stiffness * 5.0`).
3.  **Visuals:** Cyan (0), Pink (1), Gold (2).

**Acceptance Criteria:**
*   Driving Input A causes < 5% activation change in Region 1.
*   Region 2 acts as the only physical bridge.

---

### Sprint 1: Reward-Modulated Plasticity (Agency)
**Goal:** Teacher Mode, Temperature mapping, and Data Persistence.

**Implementation Guide:**
1.  **Teacher API:**
    *   `setFeedback(-1)`: Sets `meanError = 1.0` (Force High Temp).
    *   `setFeedback(1)`: Sets `meanError = 0.0` (Force Low Temp).
2.  **Temperature Logic:**
    ```typescript
    const T = Math.min(1.0, Math.max(0.0, (err - baseline) * gain));
    if (T > 0.6) v[k] += (Math.random() - 0.5) * CONSTANTS.noiseScale * T; // Agitation
    ```
3.  **Snapshot Schema:**
    ```typescript
    interface Snapshot {
      version: 1,
      timestamp: number,
      data: {
        memoryMatrix: Float32Array,
        forwardMatrix: Float32Array, // Added
        phase: Float32Array,
        spin: Float32Array
        // Do NOT save x/v (allow inference to reconstruct state)
      }
    }
    ```

**Acceptance Criteria:**
*   "Wrong" button visibly increases velocity variance.
*   "Correct" button halts motion and triggers console log "Snapshot Saved".

---

### Sprint 2: Temporal Dynamics (Prediction)
**Goal:** Asymmetric coupling and STDP using wall-clock time.

**Implementation Guide:**
1.  **Drive:** `drive = couplingStrength * delayedActivation[j]`.
2.  **STDP:**
    ```typescript
    // Must use performance.now() / 1000.0 for seconds
    const dt = lastActiveTime[j] - lastActiveTime[i]; 
    if (dt > 0 && dt < CONSTANTS.stdpWindow) {
       forwardMatrix[j*count + i] += plasticityEffective;
    }
    ```

**Acceptance Criteria:**
*   Train A->B.
*   Stimulate A. B forms in Region 2 with > 80% overlap without external drive.

---

### Sprint 3: Quantum Logic (Computation)
**Goal:** Detector clusters and hysteresis-based logic.

**Implementation Guide:**
1.  **Detector Selection:** Calculate the spatial centroid of Region 2. Select the nearest 10% of Region 2 particles to this centroid. This ensures deterministic placement at the topological center of the associative field.
2.  **Logic:**
    ```typescript
    // Complex Amplitude
    const amp = Math.sqrt(real*real + imag*imag);
    // Hysteresis
    if (amp > CONSTANTS.ampThresholdOn) activation[k] = 1.0;
    else if (amp < CONSTANTS.ampThresholdOff) activation[k] = 0.0;
    ```

**Acceptance Criteria:**
*   Visual XOR truth table matches (00->0, 01->1, 10->1, 11->0).

---

### Sprint 4: Optimization & Scaling
**Goal:** Scale to N > 2000.

**Implementation Guide:**
1.  **Spatial Index:** 
    *   Use Uniform Grid with `cell_size = CONSTANTS.gridCellSize`.
    *   Rebuild neighbors every `CONSTANTS.spatialRefreshRate` frames.
    *   **WebWorker:** Offload physics. Post `Matrix4` array back to main thread.

**Acceptance Criteria:**
*   FPS > 30 at N = 2000.

---

## 4. Testing, Telemetry & Safety Strategy

### 4.1 Deterministic Testing
To ensure reproducible CI runs:
*   **RNG:** Use a seedable generator (e.g., mulberry32) for particle initialization and noise.
*   **Time:** Mock `performance.now` in test harnesses.

### 4.2 Automated Safety Protocols (The "Circuit Breaker")
In `useFrame`:
1.  Push `plasticityRate` to `plasticityHistory`.
2.  **Rule:** If `avg(plasticityHistory)` > 0.25 for 60 frames (1 sec):
    *   Action: Force `Temperature *= 0.5` (Cool Down).
    *   Log: "Safety Trigger: Runaway Plasticity detected."

### 4.3 Automated Tests
1.  **Region Enforcement:** Region 1 activation < 0.1 when driving Region 0.
2.  **Snapshot Integrity:** Save -> Load -> Checksum match of matrices.
3.  **Logic Gate Truth Table:** 4/4 correct states.

---

## 5. Agent Reporting Template

For each PR/Task:
1.  **Summary:** One-line description.
2.  **Files Changed:** List.
3.  **Code Diff Summary:** Key logic.
4.  **Test Results:** Pass/Fail + Metrics.
5.  **Telemetry Snapshot:** `meanError`, `Temperature`, `plasticityRate`, `FPS`.
6.  **Next Recommended Task.**
