# Roadmap: From Holographic Storage to Quantum Predictive Coding AI

**Status:** Draft 1.0  
**Target:** Transform current Attractor Network (Memory) into a Computational Inference Engine (AI).

---

## Abstract

The current implementation acts as a **Holographic Memory Store** (Auto-Associative Network). It uses free energy minimization to "snap" particles into learned configurations based on global equilibrium states. To evolve this into **Artificial Intelligence**, we must transition to a **Hetero-Associative** architecture capable of logic, temporal prediction, and self-supervised learning.

This roadmap outlines four critical phases to achieve "Quantum AI" behavior using the existing particle-physics engine.

---

## Phase 1: Semantic Layering (Architecture & Morphology)

**Goal:** Break the monolithic particle cloud into functional "lobes" (Cortical Regions) to enable specific input-output processing.

### Concept
Currently, all 800 particles act as one brain region. To perform logic (e.g., "1 + 1 = 2"), we must spatially segregate them into inputs and outputs. The "Computation" happens in the physical interactions between these regions.

### Technical Specifications

#### 1. Data Structure Updates (`ParticleData`)
We need to assign a `regionID` to every particle to define its role.
```typescript
interface ParticleData {
  // ... existing fields
  regionID: Uint8Array; // 0=Input A, 1=Input B, 2=Computation Cloud
}
```

#### 2. Spatial partitioning
For $N=800$ particles:
*   **Region 0 (Indices 0-199):** Left Input (Visual Cortex A).
*   **Region 1 (Indices 200-399):** Right Input (Visual Cortex B).
*   **Region 2 (Indices 400-799):** Associative Cloud (Prefrontal Cortex/Output).

#### 3. The "Cable" Implementation
We must modify the physics loop (`App.tsx`) to enforce strict connection rules:
*   **Intra-Region (0-0, 1-1):** High Stiffness. These hold the input shapes rigidly.
*   **Inter-Region (0-2, 1-2):** "Axonal Cables." These are the learnable weights.
*   **Inhibition (0-1):** Input A and Input B should *not* interact directly. They must communicate *through* Region 2.

### Success Metric
The system allows the user to input "A" and "B" separately. Region 2 initially remains chaotic but eventually forms a distinct shape "C" that represents the combination of A and B.

---

## Phase 2: Temporal Dynamics (Sequence Learning)

**Goal:** Enable the network to predict future states based on past states (Time-series prediction).

### Concept
Current bonds are springs ($F_{AB} = -F_{BA}$). This is static. To learn sequences (e.g., spelling "H-E-L-L-O"), we need **Asymmetric Coupling**, where Particle A triggers Particle B, but B does not immediately pull back on A.

### Technical Specifications

#### 1. Directed Interaction Matrix
The `memoryMatrix` currently stores scalar distance $r_0$. We need to encode directionality.
*   **Idea:** Use the sign bit or split into two matrices (`forwardMatrix` and `feedbackMatrix`).
*   **Logic:** If $M_{ij} > 0$, $i \to j$. If $M_{ji} \approx 0$, it is a directed edge.

#### 2. Activation Wave Propagation (Delay Differential Equation)
Information must travel physically across the cloud.
```typescript
// Inside Physics Loop
// currentActivation[i] is the energy state
// delayedActivation[i] acts as the "Memory Trace"

// Force calculation:
const drive = couplingStrength * delayedActivation[j];
currentActivation[i] += drive;

// Update Delay Buffer
delayedActivation[i] = lerp(delayedActivation[i], currentActivation[i], 0.1);
```

#### 3. Sequence Training Mode
Instead of static images, we feed a stream of targets:
`Target(t0) = "A"` -> `Target(t1) = "B"` -> `Target(t2) = "C"`.
Plasticity is only enabled for bonds connecting currently active nodes to *recently* active nodes (Spike-Timing-Dependent Plasticity - STDP).

### Success Metric
After training on "A -> B", stimulating the network with "A" automatically triggers the formation of "B" in the absence of external target forces.

---

## Phase 3: Logic Gates via Wave Interference (Quantum Computing)

**Goal:** Perform boolean logic using the vibrational phase $\phi$ (Simulated Quantum Effects).

### Concept
We track a phase angle $\phi$ for every particle. By utilizing **Constructive Interference** (phases align) and **Destructive Interference** (phases oppose), we can build XOR, AND, and OR gates without digital transistors.

### Technical Specifications

#### 1. Phase Locking Inputs
*   **Logic 1:** Force Region 0 particles to $\phi = 0$.
*   **Logic 0:** Force Region 0 particles to $\phi = \pi$ (180 degrees).

#### 2. Detector Nodes (Superposition)
Designate a cluster in Region 2 as the "Detector." The activation of these particles is no longer just spatial stress, but wave amplitude:

$$ Amplitude_i = \left| \sum_{j \in neighbors} e^{i \phi_j} \right| $$

*   **IN-PHASE ($0 + 0$):** Amplitude = 2.0 (High Energy -> Logic 1).
*   **OUT-OF-PHASE ($0 + \pi$):** Amplitude = 0.0 (Zero Energy -> Logic 0).

#### 3. XOR Gate Implementation
The hardest gate for single-layer networks is XOR.
*   Config: A and B connect to Output.
*   If A=1, B=0 -> Signal passes.
*   If A=0, B=1 -> Signal passes.
*   If A=1, B=1 -> Signals interfere destructively -> Output is 0.

### Success Metric
The cloud successfully computes an XOR operation purely through vibrational dynamics, visualized by color (Red/Blue phase interference).

---

## Phase 4: Reward-Modulated Plasticity (Reinforcement Learning)

**Goal:** Give the AI "Agency" to learn tasks without hardcoded answers.

### Concept
Currently, `plasticity` is a manual toggle. We will automate this using a global **Free Energy / Reward Signal**. The network "boils" (high temp) when wrong and "freezes" (low temp) when right.

### Technical Specifications

#### 1. The Error Function (The Critic)
We need a way to grade the output without forcing positions.
```typescript
function calculateError(outputRegion: Float32Array, idealShape: Float32Array) {
   // Hausdorff Distance or simple Centroid Matching
   return distance;
}
```

#### 2. Simulated Annealing (Dopamine Control)
Introduce a global variable `Temperature` ($T$).
*   **High Error:** $T \to 1.0$.
    *   Inject random velocity noise (Agitation).
    *   Increase Plasticity ($\eta$ high). Bonds break and reform rapidly.
*   **Low Error:** $T \to 0.0$.
    *   Reduce noise.
    *   Decrease Plasticity ($\eta \to 0$). Bonds harden.

#### 3. Hebbian Consolidation
When $T$ drops below a threshold (success), the current `memoryMatrix` is snapshot-saved to long-term storage ("Skill Acquisition").

### Success Metric
We present "1 + 1" to the inputs. We grade the output. Initially, the cloud is random. Over 30 seconds of "Hot/Cold" feedback, the cloud self-organizes into a stable "2" shape to minimize the system temperature.

---

## Recommendations for Immediate Implementation

The most practical path to demonstrating "Intelligence" rather than just "Physics" is a hybrid of **Phase 1** and **Phase 4**.

**The "Teaching Simulation" Build:**
1.  Implement **Semantic Layering** (Inputs vs. Computation Cloud).
2.  Implement **Reward-Modulated Plasticity**.
3.  **User Experience:** The user acts as the teacher.
    *   User presses "Correct": System freezes weights (Dopamine hit).
    *   User presses "Wrong": System heats up and scrambles (Cortical agitation).
    
This creates a tangible, interactive AI that clearly demonstrates the Free Energy Principle in real-time.
