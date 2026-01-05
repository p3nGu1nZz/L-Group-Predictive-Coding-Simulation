

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
    },
    logic: {
        title: "Experiment D: Quantum Logic Gate",
        summary: "Demonstrates non-linear computation (XOR Gate) using Hysteresis and Destructive Interference.",
        importance: "Linear networks cannot solve XOR. By utilizing the wave nature of our particles, we achieve Destructive Interference (Phase Cancellation) to implement logic gates without silicon.",
        demonstration: "Region A (Left) and Region B (Right) are inputs. Region Center is Output. \nLogic: (A=1, B=0) -> ON. (A=0, B=1) -> ON. (A=1, B=1) -> OFF (Cancellation).",
        steps: [
            "Toggle Input A (Left) ON. Observe the center region glow (Constructive).",
            "Toggle Input B (Right) ON. Observe the center region glow.",
            "Toggle BOTH ON. Observe the center region fade (Destructive Interference).",
            "This proves the system functions as a morphological XOR gate."
        ]
    },
    adder: {
        title: "Experiment E: CPU Building Blocks",
        summary: "A suite of circuits demonstrating how particle waves can form a CPU: Half Adder, Full Adder, and Register Memory.",
        importance: "Computers are built from two things: Logic (Adders) and Memory (Registers). This experiment proves our particle system can do both.",
        demonstration: "Select a circuit mode. 'Half Adder' adds 2 bits. 'Full Adder' adds 3 bits. 'Register' stores a bit using a circular particle trap.",
        steps: [
            "Select 'FULL ADDER'. Toggle Inputs A, B, and Carry. Verify the Sum and Carry outputs.",
            "Select 'REGISTER (LATCH)'. This is a circular memory loop.",
            "Click 'WRITE 1'. Watch particles spin indefinitely (stored energy).",
            "Click 'CLEAR'. Watch the friction stop the particles (erasing memory)."
        ]
    },
    prime: {
        title: "Experiment F: Prime Search Program",
        summary: "A complex integrated circuit that increments a counter and filters for prime numbers (2, 3, 5, 7, 11, 13).",
        importance: "This combines all previous modules (Registers, Logic, Latch) into a functional program. It demonstrates how simple physical rules can be chained to perform algorithmic tasks.",
        demonstration: "The system iterates from 0 to 15. The 'Logic Mesh' (Center) acts as a physical filter. If the number is Prime, the particles are guided to the 'Result Latch' (Right) via constructive interference.",
        steps: [
            "Click 'RUN PROGRAM' to start the search loop.",
            "Observe the Counter (Left) incrementing.",
            "Watch the central Mesh. When a Prime (e.g., 5) is found, a massive energy surge travels to the Output Latch.",
            "The Output Latch turns GOLD for Prime, or dark RED for Composite."
        ]
    }
};