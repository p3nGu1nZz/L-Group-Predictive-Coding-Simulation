# Sprint 1 Status: Temporal Dynamics & Active Agency

**Date:** 2025-05-22
**Milestone:** Feature Complete (Refined UX)

---

## 1. Executive Summary

We have evolved the simulation into an **Active Inference Engine**. Sprint 1 focuses on two critical cognitive capabilities: **Agency** (thermodynamic searching) and **Prediction** (temporal association). 

We have replaced the abstract "Stress Tests" with two interactive experiments designed to demonstrate these principles intuitively.

---

## 2. Refined Experiments

### Experiment A: Active Inference (Thermodynamics)
*   **Concept:** The "Annealing" process. The system attempts to resolve a noisy input into a clear memory.
*   **Mechanism:** The user acts as the **Variational Free Energy** controller.
    *   **Agitate (High Temp):** Injects entropy (`v += noise`) to shake the system out of "false" local minima (incorrect shapes).
    *   **Settle (Low Temp):** Increases damping, allowing the system to fall into the nearest attractor basin.
*   **Goal:** Guide the system from a high-entropy "Gas" state to a low-entropy "Crystal" state (the target word "ORDER").

### Experiment B: Causal Prediction (Hebbian Time)
*   **Concept:** Learning cause-and-effect ($A \rightarrow B$).
*   **Mechanism:** Spike-Timing-Dependent Plasticity (STDP).
    *   **Training:** The system flashes "TICK" (Left) followed by "TOCK" (Right). The delay buffer records the temporal offset.
    *   **Inference:** Triggering "TICK" sends a directed energy wave through the newly formed synapses, causing "TOCK" to manifest phantom-like in the right region without external input.
*   **Visuals:** Synapses flash **White** when transmitting predictive error forward in time.

---

## 3. Technical Implementation Details

### Temporal Asymmetry
*   **Axonal Delay:** Implemented a smoothing buffer (`delayedActivation`) on particle output.
*   **Directed Force:** Unlike the symmetric structural bonds, predictive bonds are one-way (`forwardMatrix`).

### Thermodynamic Agency
*   **Feedback Loop:** User input directly modulates the global `temperature` variable, which scales the magnitude of the Brownian motion vector in the physics integrator.

---

## 4. Next Steps (Sprint 2)
*   **Quantum Logic:** Using the stable predictive states from Sprint 1 to implement `XOR` gates via hysteresis thresholds.
