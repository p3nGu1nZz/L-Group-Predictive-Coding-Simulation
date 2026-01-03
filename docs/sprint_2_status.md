# Sprint 2 Status: Quantum Logic & Computational Morphology

**Date:** 2025-05-23
**Milestone:** IN PROGRESS

---

## 1. Objective

Currently, the system can **store** patterns (Sprint 0) and **predict** temporal sequences (Sprint 1). However, it behaves linearly—it accumulates signal. To perform true computation, it must handle non-linear operations (e.g., $A + B \neq 2$, but rather $A + B = 0$ in the case of XOR).

Sprint 2 focuses on implementing **Hysteresis** (memory latching) and **Destructive Interference** to create logic gates within the particle cloud.

---

## 2. Planned Features

### A. Hysteresis (The "Latching" Effect)
*   **Problem:** Particles currently flicker on/off linearly with input.
*   **Solution:** Implement a **Schmitt Trigger** mechanism.
    *   **Activation Threshold:** High energy required to turn ON.
    *   **Deactivation Threshold:** Low energy required to turn OFF.
*   **Effect:** This stabilizes calculations, allowing the cloud to "decide" on a state and hold it even if the input fluctuates.

### B. Experiment C: The Quantum Logic Gate
*   **Goal:** Replace the current "L-Group" demo (or enhance it) to become a functional **XOR Gate**.
*   **Setup:**
    *   **Region A:** Input 1
    *   **Region B:** Input 2
    *   **Region C (Center):** The Output Processor.
*   **Dynamics:**
    *   If A or B fires -> Constructive Interference -> C turns ON.
    *   If A and B fire -> Destructive Interference (Phase Cancellation) -> C turns OFF.
    *   This effectively mimics a Quantum CNOT or Classical XOR gate.

### C. Truth Table Visualization
*   A real-time HUD element showing the state of Inputs vs Output to verify logical correctness during the simulation.

---

## 3. Technical Implementation Plan

1.  **Modify `types.ts`:** Add `hysteresisState` to `ParticleData`.
2.  **Update `App.tsx` (Physics Loop):**
    *   Implement the hysteresis check inside the loop.
    *   Implement "Phase Cancellation" logic where signals from different regions can subtract from each other based on phase alignment.
3.  **Update `UIOverlay`:**
    *   Build the `TruthTable` component.
    *   Update the Experiment C (Paper) mode to be "Logic Gate" mode.

---

## 4. Current Blockers / Risks
*   **Phase Tuning:** Getting particles to perfectly cancel out (Destructive Interference) in a noisy 3D simulation is difficult. We may need to force hard-coded phase offsets for the XOR demonstration.
