
export interface InfoContent {
    title: string;
    summary: string;
    importance: string;
    demonstration: string;
    steps: string[];
}

export const EXPERIMENT_INFO: Record<string, InfoContent> = {
    standard: {
        title: "Standard Simulation",
        summary: "A baseline sandbox environment demonstrating the core principles of the Predictive Morphology engine.",
        importance: "Establishes the fundamental physics of the Free Energy Principle, where particles self-organize to minimize the difference between their current state and a target configuration. This mimics how the brain resolves sensory ambiguity.",
        demonstration: "You will observe the spontaneous emergence of structure from chaos through energy minimization dynamics.",
        steps: [
            "Click 'INITIALIZE EXPERIMENT' to start the sequence.",
            "Step through the phases (Entropy -> Observation -> Encoding -> Recall).",
            "Observe how the system reconstructs the input 'QUANTUM' from a scrambled state using the stored memory matrix."
        ]
    },
    inference: {
        title: "Experiment A: Active Inference",
        summary: "Demonstrates 'Thermodynamic Agency', where an agent actively acts upon the world to minimize surprise.",
        importance: "Standard AI is passive (it just receives data). Biological agents are active (they sample the world). This experiment mimics biological agency by using thermodynamic temperature as an action policy to escape local minima.",
        demonstration: "You play the role of the 'Teacher'. The system tries to form the pattern 'ORDER', but it will get stuck in incorrect shapes. You must inject energy to correct it.",
        steps: [
            "Observe the particles. If they settle into a messy or incorrect shape, click and hold 'AGITATE' (Fire) to heat the system.",
            "The heat increases randomness, breaking the bad structure.",
            "When the particles start resembling 'ORDER', release the button to let them cool and settle.",
            "Use 'FREEZE' (Ice) to lock the structure in place once it looks perfect."
        ]
    },
    temporal: {
        title: "Experiment B: Causal Prediction",
        summary: "A simulation of Hebbian Temporal Learning, enabling the system to learn cause-and-effect relationships across time.",
        importance: "Static memories are insufficient for survival. An intelligent system must predict 'what happens next'. This module introduces axonal delays and asymmetric synaptic weights to encode time.",
        demonstration: "The system will learn that 'TICK' (Cause) is always followed by 'TOCK' (Effect). After training, seeing 'TICK' alone will cause the system to hallucinate 'TOCK' in the future.",
        steps: [
            "Click '1. IMPRINT' to start the training sequence. Watch the system learn the timing between the two words.",
            "Wait for the system to stabilize (Status: STABLE) and for the training cycles to complete.",
            "Click '2. TRIGGER' to flash 'TICK'.",
            "Observe Region B (Right). 'TOCK' will appear as a 'ghost' prediction without any external input, driven solely by the learned causal link."
        ]
    },
    paper: {
        title: "Experiment C: L-Group Dynamics",
        summary: "A visualization of the Lie Group (L-Group) framework applied to Predictive Coding.",
        importance: "This extends standard Predictive Coding Networks (PCN) by treating neurons as harmonic oscillators with intrinsic spin states (+1/2, -1/2). This allows for complex synchronization phenomena found in cortical waves.",
        demonstration: "You are visualizing vibrational coupling. Particles only connect and bind if their Phase and Spin align according to the L-Group math (Eq. 30).",
        steps: [
            "Adjust 'Spin Coupling' to see how likely opposite-spin particles are to interact.",
            "Adjust 'Phase Coupling' to control how much vibrational phase dictates connection strength.",
            "Toggle 'L-GROUP' input to see how the lattice self-organizes under these constraints."
        ]
    }
};
