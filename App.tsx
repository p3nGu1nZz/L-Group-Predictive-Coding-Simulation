import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Cylinder, Grid, Line, Text } from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Noise, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { SimulationParams, ParticleData, DEFAULT_PARAMS, MemoryAction, CONSTANTS, TestResult, SystemStats, MemorySnapshot, TelemetryFrame } from './types';
import { EXPERIMENT_INFO } from './Info';

// --- ParticleSystem ---
// Workaround for missing JSX types in current environment
const InstancedMesh = 'instancedMesh' as any;
const SphereGeometry = 'sphereGeometry' as any;
const LineSegments = 'lineSegments' as any;
const LineBasicMaterial = 'lineBasicMaterial' as any;
const MeshBasicMaterial = 'meshBasicMaterial' as any;
const MeshStandardMaterial = 'meshStandardMaterial' as any;

const TEMP_OBJ = new THREE.Object3D();
const TEMP_COLOR = new THREE.Color();
const TEMP_EMISSIVE = new THREE.Color();
const WHITE = new THREE.Color(1, 1, 1);
const GOLD = new THREE.Color("#FFD700");
const RED = new THREE.Color("#FF0000");
const CYAN = new THREE.Color("#00FFFF");
const MAGENTA = new THREE.Color("#FF00FF");
const GREEN = new THREE.Color("#00FF00");
const DARK = new THREE.Color("#111111");

// Paper Colors for Spin
const SPIN_UP_COLOR = new THREE.Color("#ff0055"); // Spin +1/2
const SPIN_DOWN_COLOR = new THREE.Color("#0055ff"); // Spin -1/2

// Optimization: Singleton Canvas
let sharedTextCanvas: HTMLCanvasElement | null = null;
let sharedTextCtx: CanvasRenderingContext2D | null = null;

// Paper Math: Dynamic Tanh (Eq 15)
const DyT = (x: number, alpha2: number, alpha3: number) => {
    return alpha2 * Math.tanh(alpha3 * x);
};

// Adaptive Text Sampling
const textToPoints = (text: string, targetCount: number): { positions: Float32Array, count: number } => {
  if (!text) return { positions: new Float32Array(0), count: 0 };

  if (!sharedTextCanvas) {
    sharedTextCanvas = document.createElement('canvas');
    sharedTextCanvas.width = 1024; 
    sharedTextCanvas.height = 512;
    sharedTextCtx = sharedTextCanvas.getContext('2d', { willReadFrequently: true });
  }

  const ctx = sharedTextCtx;
  if (!ctx) return { positions: new Float32Array(0), count: 0 };

  const width = sharedTextCanvas!.width;
  const height = sharedTextCanvas!.height;

  // Clear
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);
  
  // Font Sizing
  const fontSize = 95; 
  ctx.fillStyle = 'white';
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, height / 2);

  const imgData = ctx.getImageData(0, 0, width, height);
  const points: number[] = [];
  
  // Adaptive step calculation
  let pixelCount = 0;
  for(let i=0; i<imgData.data.length; i+=4) {
      if(imgData.data[i] > 128) pixelCount++;
  }
  
  const step = Math.max(2, Math.floor(Math.sqrt(pixelCount / targetCount)));
  const scale = 0.08; 

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const index = (y * width + x) * 4;
      if (imgData.data[index] > 128) {
        const jx = (Math.random() - 0.5) * step * 0.5;
        const jy = (Math.random() - 0.5) * step * 0.5;
        
        const px = (x - width / 2 + jx) * scale;
        const py = -(y - height / 2 + jy) * scale;
        const pz = 0; 
        points.push(px, py, pz);
      }
    }
  }
  
  return { positions: new Float32Array(points), count: points.length / 3 };
};

interface ParticleSystemProps {
  params: SimulationParams;
  dataRef: React.MutableRefObject<ParticleData>;
  statsRef: React.MutableRefObject<SystemStats>;
  started: boolean;
  spatialRefs: React.MutableRefObject<{
      neighborList: Int32Array;
      neighborCounts: Int32Array;
      gridHead: Int32Array;
      gridNext: Int32Array;
      frameCounter: number;
  }>;
  teacherFeedback: number; 
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef, statsRef, started, spatialRefs, teacherFeedback }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const outlineRef = useRef<THREE.InstancedMesh>(null); 
  const linesRef = useRef<THREE.LineSegments>(null);

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  const data = dataRef;
  const flashRef = useRef<Float32Array | null>(null);

  // Initialization
  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    spatialRefs.current.gridNext = new Int32Array(count);
    spatialRefs.current.neighborList = new Int32Array(count * 128); 
    spatialRefs.current.neighborCounts = new Int32Array(count);

    // Initialize Flash Buffer
    if (!flashRef.current || flashRef.current.length !== count) {
        flashRef.current = new Float32Array(count);
    }

    if (data.current.x.length !== count * 3) {
        data.current = {
            x: new Float32Array(count * 3),
            v: new Float32Array(count * 3),
            phase: new Float32Array(count),
            spin: new Int8Array(count),
            activation: new Float32Array(count),
            target: new Float32Array(count * 3),
            hasTarget: new Uint8Array(count),
            memoryMatrix: new Float32Array(memorySize).fill(-1), 
            regionID: new Uint8Array(count),
            forwardMatrix: new Float32Array(memorySize), 
            feedbackMatrix: new Float32Array(memorySize),
            delayedActivation: new Float32Array(count),
            lastActiveTime: new Float32Array(count),
            hysteresisState: new Uint8Array(count),
        };
        
        const phi = Math.PI * (3 - Math.sqrt(5)); 
        for (let i = 0; i < count; i++) {
            const y = 1 - (i + 0.5) * (2 / count); 
            const radiusAtY = Math.sqrt(1 - y * y); 
            const theta = phi * i; 
            const r = 25.0 * Math.cbrt(Math.random()); 
            
            data.current.x[i * 3] = Math.cos(theta) * radiusAtY * r;
            data.current.x[i * 3 + 1] = y * r;
            data.current.x[i * 3 + 2] = Math.sin(theta) * radiusAtY * r;

            data.current.v[i * 3] = (Math.random() - 0.5) * 1.0;
            data.current.v[i * 3 + 1] = (Math.random() - 0.5) * 1.0;
            data.current.v[i * 3 + 2] = (Math.random() - 0.5) * 1.0;

            data.current.phase[i] = Math.random() * Math.PI * 2;
            
            // Paper: Assign Intrinsic Spin (+1/2 or -1/2)
            data.current.spin[i] = Math.random() > 0.5 ? 1 : -1;

            if (i < Math.floor(count * 0.25)) data.current.regionID[i] = 0;      
            else if (i < Math.floor(count * 0.5)) data.current.regionID[i] = 1;  
            else data.current.regionID[i] = 2;                                   
        }
        memoryBank.current.clear();
    }
  }, [params.particleCount]);

  // Memory Action Handler
  useEffect(() => {
    const { type, slot, triggerId } = params.memoryAction;
    if (type === 'idle' || triggerId === 0) return;

    if (type === 'save') {
        const snapshot: MemorySnapshot = {
            x: new Float32Array(data.current.x),
            regionID: new Uint8Array(data.current.regionID),
            forwardMatrix: new Float32Array(data.current.forwardMatrix) 
        };
        memoryBank.current.set(slot, snapshot);
        console.log(`[MEMORY] Saved state to Slot ${slot}`);
    } 
    else if (type === 'load') {
        if (slot === -2) {
             memoryBank.current.clear();
             data.current.forwardMatrix.fill(0); 
        }
        else {
            const snapshot = memoryBank.current.get(slot);
            if (snapshot) {
                data.current.target.set(snapshot.x);
                data.current.hasTarget.fill(1);
                data.current.v.fill(0);
                if (snapshot.forwardMatrix) {
                    data.current.forwardMatrix.set(snapshot.forwardMatrix);
                }
            }
        }
    }
  }, [params.memoryAction.triggerId]);

  // Text / Layout Processing
  useEffect(() => {
    if (params.paused) return; 
    
    // --- EXPERIMENT D/E: INTERFEROMETER APPARATUS ---
    if (params.logicMode) {
        const count = params.particleCount;
        data.current.hasTarget.fill(1);
        data.current.activation.fill(0);
        data.current.v.fill(0);
        
        // --- EXPERIMENT E LAYOUTS ---
        if (params.gateType === 'HALF_ADDER') {
             
             if (params.circuitMode === 'REGISTER_BANK') {
                 // 4-BIT REGISTER LAYOUT
                 // Regions 0-3: The 4 storage rings.
                 const ringSize = Math.floor(count / 4);
                 for(let i=0; i<count; i++) {
                     const bitIdx = Math.floor(i / ringSize);
                     data.current.regionID[i] = bitIdx; // 0, 1, 2, 3
                     
                     // Layout: 4 Rings in a line
                     const cx = -30 + (bitIdx * 20);
                     const cy = 0;
                     
                     // Distribute in a thick torus
                     const t = (i % ringSize) / ringSize;
                     const theta = t * Math.PI * 2 * 3; // 3 windings
                     const rBase = 8;
                     const r = rBase + (Math.random()-0.5)*3;
                     
                     data.current.target[i*3] = cx + Math.cos(theta)*r;
                     data.current.target[i*3+1] = cy + Math.sin(theta)*r;
                     data.current.target[i*3+2] = (Math.random()-0.5)*4;
                     
                     data.current.x[i*3] = data.current.target[i*3];
                     data.current.x[i*3+1] = data.current.target[i*3+1];
                     data.current.x[i*3+2] = data.current.target[i*3+2];
                 }
                 return;
             }

             if (params.circuitMode === 'RIPPLE_ADDER') {
                 // 4-BIT RIPPLE CARRY ADDER
                 // Regions:
                 // 0-3: Sum Bits 0-3
                 // 4: Carry Out
                 // Layout: Horizontal Chain of 4 Adders
                 const stageCount = 4;
                 const particlesPerStage = Math.floor(count / stageCount);
                 
                 for(let i=0; i<count; i++) {
                     const stage = Math.min(stageCount-1, Math.floor(i / particlesPerStage));
                     data.current.regionID[i] = stage; // 0..3 representing Bit 0..3
                     
                     // Each stage has inputs (A,B) and output (Sum) visuals merged
                     const bx = -45 + (stage * 30);
                     const by = 0;
                     
                     // Random cloud within the block
                     const rx = (Math.random() - 0.5) * 15;
                     const ry = (Math.random() - 0.5) * 20;
                     
                     data.current.target[i*3] = bx + rx;
                     data.current.target[i*3+1] = by + ry;
                     data.current.target[i*3+2] = (Math.random()-0.5)*5;
                     
                     data.current.x[i*3] = data.current.target[i*3];
                     data.current.x[i*3+1] = data.current.target[i*3+1];
                     data.current.x[i*3+2] = data.current.target[i*3+2];
                 }
                 return;
             }

             if (params.circuitMode === 'FULL_ADDER') {
                 // FULL ADDER LAYOUT (5 Regions: A, B, Cin, Sum, Carry)
                 const partSize = Math.floor(count / 5);
                 for (let i=0; i<count; i++) {
                     let group = Math.floor(i / partSize);
                     if (group > 4) group = 4;
                     data.current.regionID[i] = group;
                     const t = (i % partSize) / partSize;
                     
                     let cx=0, cy=0;
                     // 0: A, 1: B, 2: Cin, 3: Sum, 4: Cout
                     if (group === 0) { cx = -50 + 40*t; cy = 25; }
                     else if (group === 1) { cx = -50 + 40*t; cy = 0; }
                     else if (group === 2) { cx = -50 + 40*t; cy = -25; }
                     else if (group === 3) { cx = 20 + 30*t; cy = 15; }
                     else { cx = 20 + 30*t; cy = -15; }
                     
                     // Convergence visualization
                     if (group < 3) cx += t * 10;
                     
                     data.current.target[i*3] = cx; data.current.target[i*3+1] = cy; data.current.target[i*3+2] = 0;
                     data.current.x[i*3] = cx; data.current.x[i*3+1] = cy; data.current.x[i*3+2] = 0;
                 }
                 return;
             }

             // HALF ADDER LAYOUT (Default)
             const quarterSize = Math.floor(count / 4);
             for(let i=0; i<count; i++) {
                 let group = 0;
                 if (i < quarterSize) group = 0; // Input A
                 else if (i < quarterSize * 2) group = 1; // Input B
                 else if (i < quarterSize * 3) group = 2; // Output Sum (XOR)
                 else group = 3; // Output Carry (AND)
                 
                 data.current.regionID[i] = group;
                 const t = (i % quarterSize) / quarterSize;
                 
                 let cx=0, cy=0;
                 if (group === 0) { cx = -50 + (30 * t); cy = 20; } 
                 else if (group === 1) { cx = -50 + (30 * t); cy = -20; } 
                 else if (group === 2) { cx = 20 + (30 * t); cy = 20; if (t < 0.2) cx -= 10 * (0.2-t); } 
                 else { cx = 20 + (30 * t); cy = -20; }
                 
                 cy += (Math.random() - 0.5) * 1.5;
                 const cz = (Math.random() - 0.5) * 0.5;

                 if (group === 0 || group === 1) { cx = -50 + (50 * t); cy = (group===0 ? 20 : -20) * (1-t*0.5); } 
                 else { cx = 0 + (50 * t); cy = (group===2 ? 20 : -20) * t; }
                 
                 data.current.target[i*3] = cx; data.current.target[i*3+1] = cy; data.current.target[i*3+2] = cz;
                 data.current.x[i*3] = cx; data.current.x[i*3+1] = cy; data.current.x[i*3+2] = cz;
             }
             return;
        }

        // STANDARD LOGIC GATES (3 Regions)
        const channelSize = Math.floor(count / 3);
        for(let i=0; i<count; i++) {
            let cx=0, cy=0, cz=0;
            let group = 0; 
            if (i < channelSize) group = 0; 
            else if (i < channelSize * 2) group = 1; 
            else group = 2;

            data.current.regionID[i] = group; 
            const t = (i % channelSize) / channelSize;
            
            if (group === 0) { cx = -40 + (40 * t); cy = (1-t) * 15; } 
            else if (group === 1) { cx = -40 + (40 * t); cy = -(1-t) * 15; } 
            else { cx = 40 * t; cy = 0; }
            cz = (Math.random() - 0.5) * 0.5;
            cy += (Math.random() - 0.5) * 1.5;

            data.current.target[i*3] = cx; data.current.target[i*3+1] = cy; data.current.target[i*3+2] = cz;
            data.current.x[i*3] = cx; data.current.x[i*3+1] = cy; data.current.x[i*3+2] = cz;
        }
        return;
    }

    if (!params.inputText) {
        data.current.hasTarget.fill(0);
        return;
    }

    const count = params.particleCount;
    const { positions, count: pointCount } = textToPoints(params.inputText, Math.floor(count * 0.8));
    
    data.current.hasTarget.fill(0);

    let offsetX = 0;
    if (params.targetRegion === 0) offsetX = -35;
    else if (params.targetRegion === 1) offsetX = 35;

    if (pointCount > 0) {
        const targets = [];
        for(let i=0; i<pointCount; i++) {
            targets.push({ x: positions[i*3] + offsetX, y: positions[i*3+1], z: positions[i*3+2] });
        }
        
        for (let i = targets.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [targets[i], targets[j]] = [targets[j], targets[i]];
        }
        
        const indices = Array.from({length: count}, (_, i) => i);
        for (let i = count - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const assignCount = Math.min(count, pointCount);
        
        for (let i = 0; i < assignCount; i++) {
            const pid = indices[i];
            if (params.targetRegion !== -1 && data.current.regionID[pid] !== params.targetRegion) continue;

            const t = targets[i];
            data.current.target[pid * 3] = t.x; data.current.target[pid * 3 + 1] = t.y; data.current.target[pid * 3 + 2] = t.z;
            data.current.hasTarget[pid] = 1;
            data.current.activation[pid] = 1.0;
        }
    }
  }, [params.inputText, params.particleCount, params.targetRegion, params.paused, params.logicMode, params.gateType, params.circuitMode]);

  const maxConnections = params.particleCount * 6;
  const lineGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(maxConnections * 2 * 3);
    const col = new Float32Array(maxConnections * 2 * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return geo;
  }, [maxConnections]);

  // PHYSICS LOOP
  useFrame((state, delta) => {
    if (!meshRef.current || !linesRef.current || params.paused) return;

    // --- START OF FIX ---
    const linePositions = linesRef.current.geometry.attributes.position.array as Float32Array;
    const lineColors = linesRef.current.geometry.attributes.color.array as Float32Array;
    let lineIndex = 0;
    let totalSpeed = 0;
    const delayAlpha = 0.1;
    // --- END OF FIX ---

    const effectiveChaos = started ? params.chaosMode : false;
    const timeNow = state.clock.elapsedTime;
    
    let systemTemperature = 0.0;
    if (teacherFeedback === 1) systemTemperature = 0.0; 
    else if (teacherFeedback === -1) systemTemperature = 1.0; 
    else systemTemperature = 0.05; 

    const { equilibriumDistance, stiffness, plasticity, phaseSyncRate, usePaperPhysics, spinCouplingStrength, phaseCouplingStrength, logicMode, logicState, gateType, circuitMode } = params;
    const count = params.particleCount;
    const { x, v, phase, spin, target, hasTarget, regionID, forwardMatrix, activation, delayedActivation, lastActiveTime, hysteresisState } = data.current;
    
    if (x.length === 0) return;

    const effectivePlasticity = teacherFeedback === -1 ? 0.3 : (teacherFeedback === 1 ? 0.0 : plasticity);
    const isEncoding = effectivePlasticity > 0;
    const isRecalling = params.inputText === "" && !logicMode; 

    const k_spring_base = isEncoding ? 0.2 : (isRecalling ? 1.5 : (started ? 0.2 : 0.05));
    const stiffnessMult = isRecalling ? 0.1 : (isEncoding ? 0.1 : 1.0);

    // --- LOGIC MODE: WAVE SUPERPOSITION & INTERFERENCE ---
    if (logicMode) {
        
        const A_Active = logicState[0];
        const B_Active = logicState[1];
        const C_Active = logicState[2];
        
        let phaseA = 0;
        let phaseB = Math.PI; 

        // Logic Output Calculation
        let intendedOutput = false;
        let intendedSum = false;
        let intendedCarry = false;

        // HALF ADDER LOGIC
        if (gateType === 'HALF_ADDER') {
             if (circuitMode === 'REGISTER_BANK') {
                 // Handled locally
             } else if (circuitMode === 'RIPPLE_ADDER') {
                 // Handled locally
             } else if (circuitMode === 'FULL_ADDER') {
                 // Sum = A XOR B XOR Cin
                 const xorAB = A_Active !== B_Active;
                 intendedSum = xorAB !== C_Active;
                 // Cout = (A AND B) OR (Cin AND (A XOR B))
                 intendedCarry = (A_Active && B_Active) || (C_Active && xorAB);
             } else {
                 // Half Adder
                 intendedSum = A_Active !== B_Active;
                 intendedCarry = A_Active && B_Active;
             }
        } else {
             // Standard Logic Gates
             if (gateType === 'XOR' || gateType === 'XNOR') phaseB = Math.PI; 
             else phaseB = 0; 

             switch (gateType) {
                case 'AND': intendedOutput = A_Active && B_Active; break;
                case 'OR': intendedOutput = A_Active || B_Active; break;
                case 'XOR': intendedOutput = A_Active !== B_Active; break;
                case 'NAND': intendedOutput = !(A_Active && B_Active); break;
                case 'NOR': intendedOutput = !(A_Active || B_Active); break;
                case 'XNOR': intendedOutput = A_Active === B_Active; break;
                case 'NOT': intendedOutput = !A_Active; break;
            }
        }

        // Iterate Particles
        for (let i = 0; i < count; i++) {
            const rid = regionID[i];
            
            // Hide Region 1 (Input B) particles completely if in NOT mode
            if (gateType === 'NOT' && rid === 1) {
                 TEMP_OBJ.position.set(0, -1000, 0); 
                 TEMP_OBJ.scale.set(0,0,0);
                 TEMP_OBJ.updateMatrix();
                 if(meshRef.current) meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
                 continue; 
            }

            const idx3 = i * 3;
            const px = x[idx3]; 
            
            // 1. Calculate Local Wave Amplitude
            const k = 0.3; 
            const w = 12.0; 
            const travel = k * px - w * timeNow;
            
            let localAmp = 0;
            let turbulence = 0;
            let isGolden = false;

            // Pulse Generator
            const getPulse = (phaseOffset: number) => {
                 const s = Math.sin(travel + phaseOffset);
                 return Math.pow((s + 1.0) / 2.0, 8.0); 
            };

            // --- MODE SPECIFIC PHYSICS ---
            if (circuitMode === 'REGISTER_BANK' && gateType === 'HALF_ADDER') {
                // 4-BIT REGISTER
                // Check if this bit is active
                // rid = 0..3 corresponds to bits 0..3
                const isStored = (params.registerState & (1 << rid)) !== 0;
                
                if (isStored) {
                    isGolden = true;
                    localAmp = 1.0;
                    
                    // MAGNETIC BOTTLE PHYSICS
                    const centerX = -30 + (rid * 20);
                    const centerY = 0;
                    const dx = x[idx3] - centerX;
                    const dy = x[idx3+1] - centerY;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    const radius = 8;
                    
                    // 1. Tangential Velocity (Orbital spin)
                    const spinSpeed = 0.8;
                    v[idx3] += (-dy / (dist+0.1)) * spinSpeed;
                    v[idx3+1] += (dx / (dist+0.1)) * spinSpeed;
                    
                    // 2. Strong Centripetal Force (The Bottle walls)
                    const pull = (radius - dist) * 0.2;
                    v[idx3] += (dx / (dist+0.1)) * pull;
                    v[idx3+1] += (dy / (dist+0.1)) * pull;
                    
                    // 3. Z-Axis Flattening
                    v[idx3+2] -= x[idx3+2] * 0.15;
                } else {
                    // Empty state - Scatter physics
                    // Particles should drift aimlessly or jitter
                    turbulence = 0.2; // Mild turbulence
                    localAmp = 0;
                    // Push away from center lightly to prevent clumping
                    const centerX = -30 + (rid * 20);
                    const dx = x[idx3] - centerX;
                    if (Math.abs(dx) < 2) v[idx3] += dx * 0.1;
                }
            }
            else if (circuitMode === 'RIPPLE_ADDER' && gateType === 'HALF_ADDER') {
                // 4-BIT ADDER PHYSICS
                // rid = 0..3 (Bit index)
                // Determine logic state for this bit
                const bit = rid;
                const bitA = (params.inputA_4bit >> bit) & 1;
                const bitB = (params.inputB_4bit >> bit) & 1;
                
                // Calculate Ripple Carry
                let carry = 0;
                for(let b=0; b<bit; b++) {
                    const ba = (params.inputA_4bit >> b) & 1;
                    const bb = (params.inputB_4bit >> b) & 1;
                    const sum = ba + bb + carry;
                    carry = sum > 1 ? 1 : 0;
                }
                
                const bitCin = carry;
                const bitSum = (bitA ^ bitB ^ bitCin) === 1;
                const bitCout = (bitA & bitB) | (bitCin & (bitA ^ bitB));
                
                // Visualization
                // Input activity
                if (bitA || bitB) localAmp += getPulse(0) * 0.5;
                
                // Sum activity
                if (bitSum) {
                    localAmp = getPulse(0);
                    isGolden = false; // Greenish
                }
                
                // Carry activity (visualized as turbulence or intense flow)
                if (bitCout) {
                    isGolden = true; // Gold
                    localAmp = Math.max(localAmp, getPulse(0) * 1.2);
                }
                
                // Destructive interference if inputs conflict but sum is 0
                if ((bitA && bitB && !bitCin) || (bitA && bitCin && !bitB) || (bitB && bitCin && !bitA)) {
                     // 2 inputs active, sum is 0, carry is 1. 
                     // The sum region should show interference, but Carry region carries on.
                     // Since we merge them, let's show a mix.
                     if (!bitSum) turbulence = 0.8;
                }
            }
            else if (circuitMode === 'FULL_ADDER' && gateType === 'HALF_ADDER') {
                 // FULL ADDER PHYSICS
                 if (rid === 0) { if (A_Active) localAmp = getPulse(0); }
                 else if (rid === 1) { if (B_Active) localAmp = getPulse(Math.PI); }
                 else if (rid === 2) { if (C_Active) localAmp = getPulse(0); } // Cin
                 else if (rid === 3) { // Sum
                     if (intendedSum) {
                         localAmp = getPulse(0);
                     } else if ((A_Active && B_Active && C_Active)) {
                         // 1+1+1 = 1 (Sum) + 1 (Carry). Constructive!
                         localAmp = getPulse(0);
                     } else if ((A_Active && B_Active) || (B_Active && C_Active) || (A_Active && C_Active)) {
                          // Destructive interference zone
                          turbulence = 1.0;
                     }
                 } else if (rid === 4) { // Carry
                     if (intendedCarry) {
                         localAmp = getPulse(0) * 1.5;
                         isGolden = true;
                     }
                 }
            } 
            else {
                // STANDARD HALF ADDER / GATES
                if (rid === 0) {
                    if (A_Active) localAmp = getPulse(phaseA);
                } 
                else if (rid === 1) {
                    if (B_Active) localAmp = getPulse(phaseB);
                } 
                else {
                    if (gateType === 'HALF_ADDER') {
                        // Region 2: Sum (XOR Logic)
                        if (rid === 2) {
                            if (intendedSum) localAmp = getPulse(0);
                            if (A_Active && B_Active && Math.abs(px) < 15) turbulence = 1.0;
                        } 
                        // Region 3: Carry (AND Logic)
                        else if (rid === 3) {
                             if (intendedCarry) {
                                 localAmp = getPulse(0) * 1.5; 
                                 isGolden = true;
                             }
                        }
                    } else {
                        // Standard Gate Output (Region 2)
                        const isConstructive = (A_Active && B_Active) && (gateType === 'AND' || gateType === 'OR' || gateType === 'XNOR');
                        const isBiasDriven = intendedOutput && !A_Active && !B_Active;

                        if (intendedOutput) {
                             localAmp = getPulse(0);
                             if (isConstructive) localAmp *= 1.4; 
                             if (isBiasDriven) isGolden = true;
                        }
                        
                        if (gateType === 'XOR' && A_Active && B_Active && Math.abs(px) < 15) {
                            turbulence = 1.0;
                        }
                    }
                }
            }

            // 2. State & Activation Updates
            const targetX = target[idx3];
            const targetY = target[idx3+1];
            const targetZ = target[idx3+2];
            
            // Wire Stiffness
            v[idx3] += (targetX - x[idx3]) * 0.15;
            v[idx3+1] += (targetY - x[idx3+1]) * 0.15;
            v[idx3+2] += (targetZ - x[idx3+2]) * 0.15;

            // Apply Wave Forces
            let inputActive = false;
            let outputActive = false;
            
            if (circuitMode === 'REGISTER_BANK') {
                if ((params.registerState & (1 << rid)) !== 0) outputActive = true;
            } else if (circuitMode === 'RIPPLE_ADDER') {
                // Inputs are implicit in the stage logic above, visuals driven by localAmp
                if (localAmp > 0.1) outputActive = true; 
            } else {
                if (rid < (circuitMode === 'FULL_ADDER' ? 3 : 2)) inputActive = true;
                else outputActive = true;
            }
            
            if (inputActive) {
                 // Simplify check for basic modes
                 const isActive = (rid===0 && A_Active) || (rid===1 && B_Active) || (rid===2 && circuitMode==='FULL_ADDER' && C_Active);
                 if (isActive) v[idx3+1] += localAmp * 0.15;
                 activation[i] += (isActive ? 1.0 : 0.0 - activation[i]) * 0.1;
            } else {
                 // Output Regions Logic for Flow
                 let targetAct = 0.0;
                 if (circuitMode === 'REGISTER_BANK') {
                     targetAct = ((params.registerState & (1 << rid)) !== 0) ? 1.0 : 0.0;
                 } else if (circuitMode === 'RIPPLE_ADDER') {
                     targetAct = localAmp > 0.1 ? 1.0 : 0.0;
                 } else if (gateType === 'HALF_ADDER') {
                     if (circuitMode === 'FULL_ADDER') {
                         if (rid === 3) targetAct = intendedSum ? 1.0 : 0.0;
                         if (rid === 4) targetAct = intendedCarry ? 1.0 : 0.0;
                     } else {
                         if (rid === 2) targetAct = intendedSum ? 1.0 : 0.0;
                         if (rid === 3) targetAct = intendedCarry ? 1.0 : 0.0;
                     }
                 } else {
                     targetAct = intendedOutput ? 1.0 : 0.0;
                 }
                 
                 activation[i] += (targetAct - activation[i]) * 0.1;
                 
                 // Flow visual
                 if (activation[i] > 0.1 && circuitMode !== 'REGISTER_BANK') {
                     // Flow direction: +X
                     v[idx3] += localAmp * 0.5; // Accelerate right
                     v[idx3+1] += localAmp * 0.1; // Bobble
                 }
            }
            
            // 3. XOR Destructive Interference Visuals
            if (turbulence > 0.5) {
                 const dist = Math.abs(px); 
                 const damp = Math.max(0, 1.0 - dist / 15.0);
                 
                 // Violent localized vibration without flow
                 v[idx3+1] += Math.sin(timeNow * 50.0) * damp * 2.0;
                 v[idx3+2] += Math.cos(timeNow * 40.0) * damp * 2.0; 
                 v[idx3] += (Math.random() - 0.5) * 5.0 * damp; 
            }

            // Damping (Special case for Memory Register: No Damping when active!)
            let dampingFactor = 0.8;
            if (circuitMode === 'REGISTER_BANK') {
                if ((params.registerState & (1 << rid)) !== 0) dampingFactor = 0.99; // Low friction for storage (Superconductor)
                else dampingFactor = 0.5; // High friction for clear (Insulator)
            }
            
            v[idx3] *= dampingFactor;
            v[idx3+1] *= dampingFactor;
            v[idx3+2] *= dampingFactor;

            x[idx3] += v[idx3];
            x[idx3+1] += v[idx3+1];
            x[idx3+2] += v[idx3+2];

            // 4. Render Updates
            TEMP_OBJ.position.set(x[idx3], x[idx3+1], x[idx3+2]);
            const pulseScale = activation[i] > 0.5 ? localAmp * 0.3 : 0;
            const s = 0.2 + pulseScale + activation[i] * 0.1;
            TEMP_OBJ.scale.set(s, s, s);
            TEMP_OBJ.updateMatrix();
            if(meshRef.current) meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

            // Coloring
            if (turbulence > 0.5) {
                const noise = Math.random();
                // Dark Matter Effect
                if (noise > 0.95) TEMP_COLOR.setHex(0x000000); 
                else if (noise > 0.8) TEMP_COLOR.setHex(0x330000); 
                else TEMP_COLOR.setHex(0x111111);
                
                // Occasional sparks
                if (Math.random() > 0.98) TEMP_COLOR.setHex(0xFF0000); 

            } else if (activation[i] > 0.1) {
                const brightness = activation[i];
                const pulseBoost = localAmp * 0.8; 
                const t = Math.min(1.0, brightness + pulseBoost);
                
                if (circuitMode === 'REGISTER_BANK') {
                     // Memory Plasma Color
                     TEMP_COLOR.setHex(0xFFAA00);
                     TEMP_COLOR.lerp(WHITE, localAmp);
                }
                else if (rid < 2) {
                    if (rid === 0) TEMP_COLOR.setRGB(0, t, t); // Cyan
                    else TEMP_COLOR.setRGB(t, 0, t); // Magenta
                }
                else {
                    if (isGolden) {
                         // Output Signal
                         TEMP_COLOR.setHex(0xFFD700); 
                         TEMP_COLOR.lerp(WHITE, localAmp * 0.6);
                    } else if (gateType === 'HALF_ADDER') {
                         // Sum Output (Green)
                         TEMP_COLOR.setRGB(0.1, t, 0.2);
                         if (localAmp > 0.5) TEMP_COLOR.lerp(WHITE, localAmp * 0.4);
                    } else {
                         // Generic Output
                         TEMP_COLOR.setRGB(0.1, t, 0.2); 
                         if (localAmp > 0.5) TEMP_COLOR.lerp(WHITE, localAmp * 0.5);
                    }
                }
                
                if (pulseBoost > 0.6) TEMP_COLOR.lerp(WHITE, (pulseBoost - 0.6) * 2.0);

            } else {
                TEMP_COLOR.setHex(0x111111);
            }
            if(meshRef.current) meshRef.current.setColorAt(i, TEMP_COLOR);
            
            if(outlineRef.current) {
                outlineRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
                if (turbulence > 0.5) TEMP_EMISSIVE.setRGB(0.2, 0, 0);
                else TEMP_EMISSIVE.copy(TEMP_COLOR).multiplyScalar(1.5 + localAmp * 2.0);
                outlineRef.current.setColorAt(i, TEMP_EMISSIVE);
            }
        }
        
        if (meshRef.current) {
            meshRef.current.instanceMatrix.needsUpdate = true;
            if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
        }
        if (outlineRef.current) {
            outlineRef.current.instanceMatrix.needsUpdate = true;
            if (outlineRef.current.instanceColor) outlineRef.current.instanceColor.needsUpdate = true;
        }
        linesRef.current.geometry.setDrawRange(0, 0);
        return; 
    }
    // --- END LOGIC MODE ---


    // --- STANDARD PHYSICS (Experiments A, B, C) ---
    // Spatial Hashing Refresh
    spatialRefs.current.frameCounter++;
    const CELL_SIZE = 5.0;
    const GRID_SIZE = 4096; 
    if (spatialRefs.current.frameCounter % 3 === 0) {
        const { gridHead, gridNext, neighborList, neighborCounts } = spatialRefs.current;
        gridHead.fill(-1);
        neighborCounts.fill(0);
        for (let i = 0; i < count; i++) {
            const xi = Math.floor((x[i*3] + 500) / CELL_SIZE);
            const yi = Math.floor((x[i*3+1] + 500) / CELL_SIZE);
            const zi = Math.floor((x[i*3+2] + 500) / CELL_SIZE);
            const hash = Math.abs((xi * 73856093) ^ (yi * 19349663) ^ (zi * 83492791)) % GRID_SIZE;
            gridNext[i] = gridHead[hash];
            gridHead[hash] = i;
        }
        const maxNeighbors = 48; 
        for (let i = 0; i < count; i++) {
            const xi = Math.floor((x[i*3] + 500) / CELL_SIZE);
            const yi = Math.floor((x[i*3+1] + 500) / CELL_SIZE);
            const zi = Math.floor((x[i*3+2] + 500) / CELL_SIZE);
            let foundCount = 0;
            const offset = i * maxNeighbors;
            for (let dz = -1; dz <= 1; dz++) {
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                         if (foundCount >= maxNeighbors) break;
                         const hash = Math.abs(((xi + dx) * 73856093) ^ ((yi + dy) * 19349663) ^ ((zi + dz) * 83492791)) % GRID_SIZE;
                         let j = gridHead[hash];
                         while (j !== -1 && foundCount < maxNeighbors) {
                             if (j !== i) {
                                 const distSq = (x[j*3]-x[i*3])**2 + (x[j*3+1]-x[i*3+1])**2 + (x[j*3+2]-x[i*3+2])**2;
                                 
                                 const neighborDistLimit = 25.0; 
                                 
                                 if (distSq < neighborDistLimit) { 
                                     neighborList[offset + foundCount] = j;
                                     foundCount++;
                                 }
                             }
                             j = gridNext[j];
                         }
                    }
                }
            }
            neighborCounts[i] = foundCount;
        }
    }

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0; 
      const idx3 = i * 3;
      const ix = x[idx3], iy = x[idx3 + 1], iz = x[idx3 + 2];
      const rid = regionID[i];
      const nOffset = i * 48; 
      const nCount = spatialRefs.current.neighborCounts[i];

      // Base forces
      if (effectiveChaos) {
          fx += (Math.random() - 0.5) * 1.5;
          fy += (Math.random() - 0.5) * 1.5;
          fz += (Math.random() - 0.5) * 1.5;
          fx += -iy * 0.05; fy += ix * 0.05;
      } else if (systemTemperature > 0.5) {
          const noise = systemTemperature * 0.25; 
          fx += (Math.random() - 0.5) * noise;
          fy += (Math.random() - 0.5) * noise;
          fz += (Math.random() - 0.5) * noise;
      } else if (!started) {
          fx += (Math.random() - 0.5) * 0.02;
          fy += (Math.random() - 0.5) * 0.02;
          fz += (Math.random() - 0.5) * 0.02;
          fx += -iy * 0.001; fy += ix * 0.001;
      }

      if (hasTarget[i] && !effectiveChaos && started) {
          const dx = target[idx3] - ix;
          const dy = target[idx3+1] - iy;
          const dz = target[idx3+2] - iz;
          // Standard Mode: Loose springs for fluid motion
          const k = k_spring_base;
          
          fx += dx * k;
          fy += dy * k;
          fz += dz * k;
      }

      // --- PHYSICAL COUPLING (Universal for all modes) ---
      // This is what allows the wave from Input to propagate to Output
      
      const baseInteractionStrength = hasTarget[i] ? 0.1 : 1.0;
      let predictionLock = 0.0;

      if (baseInteractionStrength > 0.01) {
          for (let n = 0; n < nCount; n++) {
            const j = spatialRefs.current.neighborList[nOffset + n];
            const rj = regionID[j];
            
             if ((rid === 0 && rj === 1) || (rid === 1 && rj === 0)) continue;

            const dx = x[j*3] - ix; const dy = x[j*3+1] - iy; const dz = x[j*3+2] - iz;
            const distSq = dx*dx + dy*dy + dz*dz;
            if (distSq < 0.01 || distSq > 64.0) continue; 
            const dist = Math.sqrt(distSq);
            
            const r0 = equilibriumDistance; 
            let force = 0;
            
            const phaseDiff = phase[j] - phase[i];

            let paperCoupling = 1.0;
            if (usePaperPhysics) {
                const phaseTerm = (Math.cos(phaseDiff) + 1.0) / 2.0; 
                const spinTerm = 1.0 + spinCouplingStrength * spin[i] * spin[j];
                const syncStrength = DyT(phaseTerm, 1.0, 2.0);
                paperCoupling = (phaseTerm * phaseCouplingStrength) * spinTerm * syncStrength;
            }
            
            let springF = 0;
            
            if (dist < r0) springF = -stiffness * stiffnessMult * (r0 - dist) * 2.0;
            else if (!hasTarget[i]) springF = stiffness * stiffnessMult * (dist - r0) * 0.1;

            force += springF;

            // STDP & Prediction (Standard Mode Only)
            if (!usePaperPhysics) {
                const weightJI = forwardMatrix[j * count + i]; 
                if (weightJI > 0.01) {
                    const signal = weightJI * delayedActivation[j];
                    predictionLock += signal; 
                    force += signal * 0.01; 
                }
            }
            
            force *= baseInteractionStrength;
            const invDist = 1.0 / dist;
            fx += dx * invDist * force; fy += dy * invDist * force; fz += dz * invDist * force;
            
            const showLine = (j > i && dist < r0 * 2.0);

            if (showLine && lineIndex < maxConnections) {
                const li = lineIndex * 6;
                linePositions[li] = ix; linePositions[li+1] = iy; linePositions[li+2] = iz;
                linePositions[li+3] = x[j*3]; linePositions[li+4] = x[j*3+1]; linePositions[li+5] = x[j*3+2];
                
                lineColors[li] = 0; lineColors[li+1] = 0.3; lineColors[li+2] = 0.8; 
                lineColors[li+3] = 0; lineColors[li+4] = 0.3; lineColors[li+5] = 0.8;
                
                lineIndex++;
            }
          }
      }

      let particleDamping = 0.85; 
      if (usePaperPhysics) particleDamping = 0.90; 
      if (!started) particleDamping = 0.95; 
      else if (effectiveChaos) particleDamping = 0.98;
      
      v[idx3] = v[idx3] * particleDamping + fx;
      v[idx3+1] = v[idx3+1] * particleDamping + fy;
      v[idx3+2] = v[idx3+2] * particleDamping + fz;
      
      const speedSq = v[idx3]**2 + v[idx3+1]**2 + v[idx3+2]**2;
      const speed = Math.sqrt(speedSq);
      totalSpeed += speed;

      x[idx3] += v[idx3]; x[idx3+1] += v[idx3+1]; x[idx3+2] += v[idx3+2];
      
      // Boundary check
      const rSq = x[idx3]**2 + x[idx3+1]**2 + x[idx3+2]**2;
      if (rSq > 3000) { x[idx3]*=0.99; x[idx3+1]*=0.99; x[idx3+2]*=0.99; }
      
      if (!logicMode) phase[i] += phaseSyncRate * phaseDelta + 0.05;

      // --- HYSTERESIS & ACTIVATION ---
      let externalDrive = hasTarget[i] ? 1.0 : 0.0;

    
      // Standard Hysteresis Logic
      const totalInputEnergy = externalDrive; 
      const currentHysteresis = hysteresisState[i];
      let nextHysteresis = currentHysteresis;
      if (currentHysteresis === 0) {
          if (totalInputEnergy > CONSTANTS.activationThresholdHigh) nextHysteresis = 1;
      } else {
          if (totalInputEnergy < CONSTANTS.activationThresholdLow) nextHysteresis = 0;
      }
      hysteresisState[i] = nextHysteresis;
      const targetActivation = nextHysteresis === 1 ? 1.0 : 0.0;
      activation[i] += (targetActivation - activation[i]) * 0.2;
      delayedActivation[i] = delayedActivation[i] * (1 - delayAlpha) + activation[i] * delayAlpha;


      // --- VISUALIZATION UPDATE ---
      TEMP_OBJ.position.set(x[idx3], x[idx3+1], x[idx3+2]);
      
      // Standard Coloring ... (Preserved from previous logic)
      const entropy = Math.min(1.0, speed * 0.5); 
      let r=0, g=0, b=0;
      
      if (usePaperPhysics) {
          if (spin[i] > 0) {
              r = SPIN_UP_COLOR.r; g = SPIN_UP_COLOR.g; b = SPIN_UP_COLOR.b; 
          } else {
              r = SPIN_DOWN_COLOR.r; g = SPIN_DOWN_COLOR.g; b = SPIN_DOWN_COLOR.b; 
          }
          const pulse = (Math.sin(phase[i]) + 1) * 0.5;
          r += pulse * 0.3; g += pulse * 0.3; b += pulse * 0.3;
          if (activation[i] > 0.8) { r += 0.5; g += 0.5; b += 0.5; }
      } else {
          let errorMetric = 0;
          if (hasTarget[i]) {
              const dx = target[idx3] - x[idx3];
              const dy = target[idx3+1] - x[idx3+1];
              const dz = target[idx3+2] - x[idx3+2];
              errorMetric = Math.sqrt(dx*dx + dy*dy + dz*dz);
          } else {
              errorMetric = speed * 25.0; 
          }
          const t = Math.min(1.0, errorMetric / 10.0);
          TEMP_COLOR.lerpColors(GOLD, RED, t);
          r = TEMP_COLOR.r; g = TEMP_COLOR.g; b = TEMP_COLOR.b;
      }

      const coreMix = entropy * 2.0; 
      TEMP_COLOR.setRGB(r * (1+coreMix), g * (1+coreMix), b * (1+coreMix)); 
      
      if (flashRef.current && flashRef.current[i] > 0.05) {
           const f = flashRef.current[i];
           TEMP_COLOR.lerp(WHITE, Math.min(1.0, f));
           if (f > 1.0) { TEMP_COLOR.r += f * 0.5; TEMP_COLOR.g += f * 0.5; TEMP_COLOR.b += f * 0.5; }
      }
      else if (activation[i] > 0.5) {
          TEMP_COLOR.lerp(WHITE, activation[i] * 0.5);
      }
      if(meshRef.current) meshRef.current.setColorAt(i, TEMP_COLOR);
      
      
      // Update Scale/Matrix
      const s = hasTarget[i] ? 0.25 : 0.35; 
      TEMP_OBJ.scale.set(s, s, s);
      
      TEMP_OBJ.updateMatrix();
      if(meshRef.current) meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
      if(outlineRef.current) outlineRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      // Glow Update
      const pulse = 1.0 + Math.sin(phase[i]) * 0.3;
      const glowIntensity = usePaperPhysics ? 3.5 : 2.5 + (Math.min(1.0, speed * 0.5)) * 4.0;
      TEMP_EMISSIVE.setRGB(TEMP_COLOR.r * glowIntensity * pulse, TEMP_COLOR.g * glowIntensity * pulse, TEMP_COLOR.b * glowIntensity * pulse);
      
      if(outlineRef.current) outlineRef.current.setColorAt(i, TEMP_EMISSIVE);

    } // End of Particle Loop

    // --- Post-Loop Updates ---
    if (meshRef.current) {
        meshRef.current.instanceMatrix.needsUpdate = true;
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    }
    if (outlineRef.current) {
        outlineRef.current.instanceMatrix.needsUpdate = true;
        if (outlineRef.current.instanceColor) outlineRef.current.instanceColor.needsUpdate = true;
    }
    if (linesRef.current) {
        linesRef.current.geometry.setDrawRange(0, lineIndex * 2);
        linesRef.current.geometry.attributes.position.needsUpdate = true;
        linesRef.current.geometry.attributes.color.needsUpdate = true;
    }

    if (statsRef.current && count > 0) {
        statsRef.current.meanSpeed = totalSpeed / count;
    }

  });

  return (
    <>
      <InstancedMesh ref={meshRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[1, 8, 8]} />
        <MeshStandardMaterial vertexColors emissive="#444444" emissiveIntensity={0.5} roughness={0.4} metalness={0.8} />
      </InstancedMesh>
      
      <InstancedMesh ref={outlineRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[1.05, 8, 8]} />
        <MeshBasicMaterial vertexColors blending={THREE.AdditiveBlending} transparent opacity={0.3} depthWrite={false} side={THREE.BackSide} />
      </InstancedMesh>

      <LineSegments ref={linesRef} geometry={lineGeometry}>
        <LineBasicMaterial vertexColors blending={THREE.AdditiveBlending} transparent opacity={0.4} depthWrite={false} />
      </LineSegments>
    </>
  );
};

const LogicGateOverlay: React.FC<{ gateType: string, circuitMode: string }> = ({ gateType, circuitMode }) => {
  const points = useMemo(() => {
     const shape = [];
     
     if (gateType === 'HALF_ADDER') {
         if (circuitMode === 'REGISTER_BANK') {
             // 4 Rings visual
             for(let i=0; i<4; i++) {
                 const cx = -30 + (i * 20);
                 const circle = [];
                 for(let j=0; j<=32; j++) {
                     const theta = (j/32) * Math.PI * 2;
                     circle.push(new THREE.Vector3(cx + Math.cos(theta)*8, Math.sin(theta)*8, 0));
                 }
                 shape.push(circle);
             }
             return shape;
         }

         if (circuitMode === 'RIPPLE_ADDER') {
             // 4 Stages chained
             for(let i=0; i<4; i++) {
                 const bx = -45 + (i * 30);
                 const box = [
                     new THREE.Vector3(bx-10, 20, 0), new THREE.Vector3(bx+10, 20, 0),
                     new THREE.Vector3(bx+10, -20, 0), new THREE.Vector3(bx-10, -20, 0),
                     new THREE.Vector3(bx-10, 20, 0)
                 ];
                 shape.push(box);
                 // Input lines
                 shape.push([new THREE.Vector3(bx, 30, 0), new THREE.Vector3(bx, 20, 0)]);
                 // Carry line to next
                 if(i < 3) {
                     shape.push([new THREE.Vector3(bx+10, 0, 0), new THREE.Vector3(bx+20, 0, 0)]);
                 }
             }
             return shape;
         }

         if (circuitMode === 'FULL_ADDER') {
             // A
             shape.push([new THREE.Vector3(-50, 25, 0), new THREE.Vector3(0, 25, 0)]);
             // B
             shape.push([new THREE.Vector3(-50, 0, 0), new THREE.Vector3(0, 0, 0)]);
             // Cin
             shape.push([new THREE.Vector3(-50, -25, 0), new THREE.Vector3(0, -25, 0)]);
             
             // Sum
             shape.push([new THREE.Vector3(0, 25, 0), new THREE.Vector3(50, 15, 0)]);
             // Cout
             shape.push([new THREE.Vector3(0, -25, 0), new THREE.Vector3(50, -15, 0)]);
             return shape;
         }

         // Half Adder Schematic
         // Top Line (Input A -> Sum)
         shape.push([new THREE.Vector3(-50, 20, 0), new THREE.Vector3(0, 20, 0)]);
         shape.push([new THREE.Vector3(0, 20, 0), new THREE.Vector3(20, 20, 0)]);
         // Bottom Line (Input B -> Carry)
         shape.push([new THREE.Vector3(-50, -20, 0), new THREE.Vector3(0, -20, 0)]);
         shape.push([new THREE.Vector3(0, -20, 0), new THREE.Vector3(20, -20, 0)]);
         
         // Cross Connections (Visual only)
         shape.push([new THREE.Vector3(-20, 20, 0), new THREE.Vector3(-20, -20, 0)]);
         shape.push([new THREE.Vector3(-10, -20, 0), new THREE.Vector3(-10, 20, 0)]);
         
         return shape;
     }

     // Mach-Zehnder Interferometer Shape (Standard Gate)
     const topCurve = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-40, 15, 0),
         new THREE.Vector3(-20, 15, 0),
         new THREE.Vector3(0, 0, 0)
     ]);
     shape.push(topCurve.getPoints(20));

     const bottomCurve = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-40, -15, 0),
         new THREE.Vector3(-20, -15, 0),
         new THREE.Vector3(0, 0, 0)
     ]);
     shape.push(bottomCurve.getPoints(20));

     shape.push([new THREE.Vector3(0, 0, 0), new THREE.Vector3(40, 0, 0)]);

     return shape;
  }, [gateType, circuitMode]);

  return (
      <group position={[0, 0, 0]}>
          {points.map((p, i) => (
              <Line key={i} points={p} color="cyan" opacity={0.2} transparent lineWidth={1} />
          ))}
          {gateType === 'HALF_ADDER' && circuitMode === 'REGISTER_BANK' && (
              <>
                 <Text position={[-30, 12, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">BIT 0</Text>
                 <Text position={[-10, 12, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">BIT 1</Text>
                 <Text position={[10, 12, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">BIT 2</Text>
                 <Text position={[30, 12, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">BIT 3</Text>
                 <Text position={[0, -15, 0]} fontSize={2} color="#FFD700" anchorX="center" anchorY="middle">MAGNETIC REGISTER BANK (4-BIT)</Text>
              </>
          )}
          {gateType === 'HALF_ADDER' && circuitMode === 'RIPPLE_ADDER' && (
              <>
                 <Text position={[-45, 25, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">ADDER 0</Text>
                 <Text position={[-15, 25, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">ADDER 1</Text>
                 <Text position={[15, 25, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">ADDER 2</Text>
                 <Text position={[45, 25, 0]} fontSize={2} color="#FFFFFF" anchorX="center" anchorY="middle">ADDER 3</Text>
                 <Text position={[0, -25, 0]} fontSize={2} color="#FFD700" anchorX="center" anchorY="middle">4-BIT RIPPLE CARRY CHAIN</Text>
              </>
          )}
          {gateType === 'HALF_ADDER' && circuitMode === 'FULL_ADDER' && (
              <>
                <Text position={[-52, 25, 0]} fontSize={2} color="#00FFFF" anchorX="right" anchorY="middle">A</Text>
                <Text position={[-52, 0, 0]} fontSize={2} color="#FF00FF" anchorX="right" anchorY="middle">B</Text>
                <Text position={[-52, -25, 0]} fontSize={2} color="#00FF00" anchorX="right" anchorY="middle">C-IN</Text>
                <Text position={[52, 15, 0]} fontSize={2} color="#00FFFF" anchorX="left" anchorY="middle">SUM</Text>
                <Text position={[52, -15, 0]} fontSize={2} color="#FFD700" anchorX="left" anchorY="middle">C-OUT</Text>
              </>
          )}
          {gateType === 'HALF_ADDER' && circuitMode === 'HALF_ADDER' && (
              <>
                <Text position={[-52, 20, 0]} fontSize={2} color="#00FFFF" anchorX="right" anchorY="middle">A</Text>
                <Text position={[-52, -20, 0]} fontSize={2} color="#FF00FF" anchorX="right" anchorY="middle">B</Text>
                <Text position={[22, 20, 0]} fontSize={2} color="#00FF00" anchorX="left" anchorY="middle">SUM (S)</Text>
                <Text position={[22, -20, 0]} fontSize={2} color="#FFD700" anchorX="left" anchorY="middle">CARRY (C)</Text>
              </>
          )}
          {gateType !== 'HALF_ADDER' && (
              <>
                <Text position={[-42, 15, 0]} fontSize={2} color="#00FFFF" anchorX="right" anchorY="middle">A</Text>
                <Text position={[-42, -15, 0]} fontSize={2} color="#FF00FF" anchorX="right" anchorY="middle">B</Text>
                <Text position={[42, 0, 0]} fontSize={2} color="white" anchorX="left" anchorY="middle">OUT</Text>
              </>
          )}
          <Text position={[0, 30, 0]} fontSize={1.5} color="gray" anchorX="center" anchorY="middle">{circuitMode.replace('_', ' ')}</Text>
      </group>
  )
}

// --- UI Components ---

const TruthTable: React.FC<{ params: SimulationParams }> = ({ params }) => {
    const rowClass = (active: boolean) => active ? "bg-cyan-900/50 text-white font-bold border border-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.3)]" : "text-gray-600";
    
    const { gateType, circuitMode, logicState } = params;
    const a = logicState[0];
    const b = logicState[1];
    const c = logicState[2];

    if (gateType === 'HALF_ADDER') {
        if (circuitMode === 'REGISTER_BANK') return (
            <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
                <div className="text-[10px] text-gray-500 font-mono text-center">MEMORY BANK STATUS</div>
                <div className="text-2xl font-mono text-center text-amber-500 font-bold tracking-widest mt-1">
                    {params.registerState.toString(2).padStart(4, '0')}
                </div>
                <div className="text-[9px] text-gray-500 text-center">DEC: {params.registerState}</div>
            </div>
        );

        if (circuitMode === 'RIPPLE_ADDER') {
            const sum = params.inputA_4bit + params.inputB_4bit;
            return (
                 <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
                     <div className="text-[10px] text-gray-500 font-mono text-center mb-1">4-BIT ADDER STATE</div>
                     <div className="flex justify-between text-xs font-mono text-gray-400 mb-1">
                         <span>A: {params.inputA_4bit.toString(2).padStart(4,'0')} ({params.inputA_4bit})</span>
                         <span>+</span>
                         <span>B: {params.inputB_4bit.toString(2).padStart(4,'0')} ({params.inputB_4bit})</span>
                     </div>
                     <div className="border-t border-gray-700 my-1"></div>
                     <div className="text-xl font-mono text-center text-green-400 font-bold">
                         = {sum.toString(2).padStart(5,'0')} ({sum})
                     </div>
                 </div>
            )
        }

        if (circuitMode === 'FULL_ADDER') {
            const getSum = (vA: boolean, vB: boolean, vC: boolean) => ((vA !== vB) !== vC) ? 1 : 0;
            const getCarry = (vA: boolean, vB: boolean, vC: boolean) => ((vA && vB) || (vC && (vA !== vB))) ? 1 : 0;
            
            return (
                 <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
                     <div className="text-[10px] text-gray-500 font-mono mb-2 uppercase tracking-widest text-center border-b border-gray-800 pb-1">Full Adder Truth Table</div>
                     <div className="grid grid-cols-5 gap-1 text-[10px] font-mono text-center">
                        <div className="text-gray-400 font-bold pb-1">A</div>
                        <div className="text-gray-400 font-bold pb-1">B</div>
                        <div className="text-gray-400 font-bold pb-1">Cin</div>
                        <div className="text-green-500 font-bold pb-1">S</div>
                        <div className="text-yellow-500 font-bold pb-1">C</div>
                        {/* Current State Highlight */}
                        <div className={rowClass(a)}>{a?1:0}</div>
                        <div className={rowClass(b)}>{b?1:0}</div>
                        <div className={rowClass(c)}>{c?1:0}</div>
                        <div className="text-white font-bold">{getSum(a,b,c)}</div>
                        <div className="text-white font-bold">{getCarry(a,b,c)}</div>
                     </div>
                 </div>
            )
        }

        // Half Adder Truth Table
        const getSum = (vA: boolean, vB: boolean) => (vA !== vB) ? 1 : 0;
        const getCarry = (vA: boolean, vB: boolean) => (vA && vB) ? 1 : 0;
        return (
            <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
                <div className="text-[10px] text-gray-500 font-mono mb-2 uppercase tracking-widest text-center border-b border-gray-800 pb-1">Half Adder Truth Table</div>
                <div className="grid grid-cols-4 gap-1 text-xs font-mono text-center">
                    <div className="text-gray-400 font-bold pb-1">A</div>
                    <div className="text-gray-400 font-bold pb-1">B</div>
                    <div className="text-green-500 font-bold pb-1">S</div>
                    <div className="text-yellow-500 font-bold pb-1">C</div>
                    <div className={rowClass(a)}>{a?1:0}</div>
                    <div className={rowClass(b)}>{b?1:0}</div>
                    <div className="text-white font-bold">{getSum(a,b)}</div>
                    <div className="text-white font-bold">{getCarry(a,b)}</div>
                </div>
            </div>
        )
    }

    // Standard Gates Logic
    const getOut = (vA: boolean, vB: boolean) => {
        switch(gateType) {
            case 'AND': return (vA && vB) ? 1 : 0;
            case 'OR': return (vA || vB) ? 1 : 0;
            case 'XOR': return (vA !== vB) ? 1 : 0;
            case 'NAND': return !(vA && vB) ? 1 : 0;
            case 'NOR': return !(vA || vB) ? 1 : 0;
            case 'XNOR': return (vA === vB) ? 1 : 0;
            case 'NOT': return (!vA) ? 1 : 0;
            default: return 0;
        }
    }

    if (gateType === 'NOT') {
         return (
            <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
                <div className="text-[10px] text-gray-500 font-mono mb-2 uppercase tracking-widest text-center border-b border-gray-800 pb-1">Logic (NOT)</div>
                <div className="grid grid-cols-2 gap-1 text-xs font-mono text-center">
                    <div className="text-gray-400 font-bold pb-1">IN</div><div className="text-gray-400 font-bold pb-1">OUT</div>
                    <div className={rowClass(a)}>{a?1:0}</div><div className="text-white font-bold">{getOut(a, false)}</div>
                </div>
            </div>
        )
    }

    return (
        <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
            <div className="text-[10px] text-gray-500 font-mono mb-2 uppercase tracking-widest text-center border-b border-gray-800 pb-1">Logic ({gateType})</div>
            <div className="grid grid-cols-3 gap-1 text-xs font-mono text-center">
                <div className="text-gray-400 font-bold pb-1">A</div><div className="text-gray-400 font-bold pb-1">B</div><div className="text-gray-400 font-bold pb-1">OUT</div>
                <div className={rowClass(a)}>{a?1:0}</div><div className={rowClass(b)}>{b?1:0}</div><div className="text-white font-bold">{getOut(a,b)}</div>
            </div>
        </div>
    )
}

const UIOverlay: React.FC<{ 
    mode: string, 
    params: SimulationParams, 
    setParams: any, 
    setFeedback: any, 
    onExit: () => void,
    onShowInfo: () => void,
    testResults: TestResult[],
    isTesting: boolean,
    onRunTests: () => void
}> = ({ mode, params, setParams, setFeedback, onExit, onShowInfo, testResults, isTesting, onRunTests }) => {
    
    // --- Mode Specific Controls ---
    const renderControls = () => {
        if (mode === 'inference') {
            return (
                <div className="mt-4 space-y-2">
                    <div className="text-xs text-yellow-500 font-bold tracking-widest border-b border-yellow-900 pb-1">THERMODYNAMIC AGENCY</div>
                    <div className="flex gap-2">
                        <button 
                            onMouseDown={() => setFeedback(-1)} 
                            onMouseUp={() => setFeedback(0)}
                            className="flex-1 py-4 bg-red-900/30 border border-red-500/50 hover:bg-red-500/50 text-red-200 text-xs font-bold"
                        >
                            AGITATE (HEAT)
                        </button>
                        <button 
                            onMouseDown={() => setFeedback(1)} 
                            onMouseUp={() => setFeedback(0)}
                            className="flex-1 py-4 bg-cyan-900/30 border border-cyan-500/50 hover:bg-cyan-500/50 text-cyan-200 text-xs font-bold"
                        >
                            CRYSTALLIZE (COOL)
                        </button>
                    </div>
                </div>
            )
        }
        if (mode === 'logic' || mode === 'adder') {
            return (
                <div className="mt-4 space-y-2">
                    {mode === 'logic' && (
                    <>
                    <div className="text-xs text-red-500 font-bold tracking-widest border-b border-red-900 pb-1">GATE TYPE SELECTION</div>
                    <div className="grid grid-cols-4 gap-1 mb-3">
                         {['AND', 'OR', 'XOR', 'NOT'].map((g) => (
                             <button
                                key={g}
                                onClick={() => setParams((p: any) => ({...p, gateType: g, logicState: [false, false, false]}))}
                                className={`py-2 text-[10px] font-bold border transition-all ${params.gateType === g ? 'bg-red-500 text-black border-red-400' : 'bg-black text-gray-500 border-gray-700 hover:border-gray-500'}`}
                             >
                                 {g}
                             </button>
                         ))}
                    </div>
                    <div className="grid grid-cols-3 gap-1 mb-3">
                         {['NAND', 'NOR', 'XNOR'].map((g) => (
                             <button
                                key={g}
                                onClick={() => setParams((p: any) => ({...p, gateType: g, logicState: [false, false, false]}))}
                                className={`py-2 text-[10px] font-bold border transition-all ${params.gateType === g ? 'bg-red-500 text-black border-red-400' : 'bg-black text-gray-500 border-gray-700 hover:border-gray-500'}`}
                             >
                                 {g}
                             </button>
                         ))}
                    </div>
                    </>
                    )}

                    {mode === 'adder' && (
                        <>
                        <div className="text-xs text-orange-500 font-bold tracking-widest border-b border-orange-900 pb-1">COMPONENT TYPE</div>
                        <div className="grid grid-cols-4 gap-1 mb-3">
                            <button onClick={() => setParams((p: any) => ({...p, circuitMode: 'HALF_ADDER', loopActive: false, logicState: [false,false,false]}))} className={`py-2 text-[8px] font-bold border ${params.circuitMode === 'HALF_ADDER' ? 'bg-orange-500 text-black border-orange-400' : 'bg-black text-gray-500 border-gray-700'}`}>HALF ADDER</button>
                            <button onClick={() => setParams((p: any) => ({...p, circuitMode: 'FULL_ADDER', loopActive: false, logicState: [false,false,false]}))} className={`py-2 text-[8px] font-bold border ${params.circuitMode === 'FULL_ADDER' ? 'bg-orange-500 text-black border-orange-400' : 'bg-black text-gray-500 border-gray-700'}`}>FULL ADDER</button>
                            <button onClick={() => setParams((p: any) => ({...p, circuitMode: 'RIPPLE_ADDER', loopActive: false, logicState: [false,false,false]}))} className={`py-2 text-[8px] font-bold border ${params.circuitMode === 'RIPPLE_ADDER' ? 'bg-orange-500 text-black border-orange-400' : 'bg-black text-gray-500 border-gray-700'}`}>CHAIN ADDER</button>
                            <button onClick={() => setParams((p: any) => ({...p, circuitMode: 'REGISTER_BANK', loopActive: false, memoryBit: 0}))} className={`py-2 text-[8px] font-bold border ${params.circuitMode === 'REGISTER_BANK' ? 'bg-orange-500 text-black border-orange-400' : 'bg-black text-gray-500 border-gray-700'}`}>MEM BANK</button>
                        </div>
                        </>
                    )}

                    <div className="text-xs text-red-500 font-bold tracking-widest border-b border-red-900 pb-1">CIRCUIT CONTROLS</div>
                    
                    {params.circuitMode === 'REGISTER_BANK' ? (
                        <div className="space-y-2 mt-2">
                             <div className="grid grid-cols-4 gap-1">
                                <button onClick={() => setParams((p: any) => ({...p, registerState: p.registerState ^ 1}))} className={`text-[10px] border ${params.registerState&1 ? 'bg-amber-500 text-black' : 'border-gray-700'}`}>B0</button>
                                <button onClick={() => setParams((p: any) => ({...p, registerState: p.registerState ^ 2}))} className={`text-[10px] border ${params.registerState&2 ? 'bg-amber-500 text-black' : 'border-gray-700'}`}>B1</button>
                                <button onClick={() => setParams((p: any) => ({...p, registerState: p.registerState ^ 4}))} className={`text-[10px] border ${params.registerState&4 ? 'bg-amber-500 text-black' : 'border-gray-700'}`}>B2</button>
                                <button onClick={() => setParams((p: any) => ({...p, registerState: p.registerState ^ 8}))} className={`text-[10px] border ${params.registerState&8 ? 'bg-amber-500 text-black' : 'border-gray-700'}`}>B3</button>
                             </div>
                             <button 
                                onClick={() => setParams((p: any) => ({...p, registerState: 0}))}
                                className="w-full py-2 bg-red-900/30 border border-red-500 text-red-200 text-xs font-bold"
                             >
                                CLEAR ALL
                             </button>
                             <button 
                                onClick={() => setParams((p: any) => ({...p, loopActive: !p.loopActive}))}
                                className={`w-full py-2 text-xs font-bold border ${params.loopActive ? 'bg-amber-900/40 border-amber-500 text-amber-200' : 'bg-gray-900 border-gray-600'}`}
                             >
                                {params.loopActive ? "STOP TEST CYCLE" : "RUN TEST CYCLE"}
                             </button>
                        </div>
                    ) : params.circuitMode === 'RIPPLE_ADDER' ? (
                        <div className="space-y-2 mt-2">
                             <div className="flex justify-between items-center text-xs">
                                 <span className="text-cyan-400">INPUT A</span>
                                 <button onClick={() => setParams((p: any) => ({...p, inputA_4bit: (p.inputA_4bit + 1) % 16}))} className="px-2 py-1 border border-cyan-500 text-white">+</button>
                             </div>
                             <div className="flex justify-between items-center text-xs">
                                 <span className="text-pink-400">INPUT B</span>
                                 <button onClick={() => setParams((p: any) => ({...p, inputB_4bit: (p.inputB_4bit + 1) % 16}))} className="px-2 py-1 border border-pink-500 text-white">+</button>
                             </div>
                             <button 
                                onClick={() => setParams((p: any) => ({...p, loopActive: !p.loopActive}))}
                                className={`w-full py-2 text-xs font-bold border mt-2 ${params.loopActive ? 'bg-cyan-900/40 border-cyan-500 text-cyan-200' : 'bg-gray-900 border-gray-600'}`}
                             >
                                {params.loopActive ? "STOP CALCULATION LOOP" : "RUN CALCULATION LOOP"}
                             </button>
                        </div>
                    ) : (
                        <>
                        <div className="grid grid-cols-2 gap-2">
                             <button 
                                disabled={params.loopActive}
                                onClick={() => setParams((p: any) => ({...p, logicState: [!p.logicState[0], p.logicState[1], p.logicState[2]]}))}
                                className={`p-3 text-xs font-bold border transition-all ${params.logicState[0] ? 'bg-cyan-500 text-black border-cyan-400' : 'bg-black text-gray-500 border-gray-700'} ${params.loopActive ? 'opacity-50' : ''}`}
                             >
                                 INPUT A: {params.logicState[0] ? "1" : "0"}
                             </button>
                             
                             {params.gateType !== 'NOT' && (
                                 <button 
                                    disabled={params.loopActive}
                                    onClick={() => setParams((p: any) => ({...p, logicState: [p.logicState[0], !p.logicState[1], p.logicState[2]]}))}
                                    className={`p-3 text-xs font-bold border transition-all ${params.logicState[1] ? 'bg-pink-500 text-black border-pink-400' : 'bg-black text-gray-500 border-gray-700'} ${params.loopActive ? 'opacity-50' : ''}`}
                                 >
                                     INPUT B: {params.logicState[1] ? "1" : "0"}
                                 </button>
                             )}
                             
                             {params.circuitMode === 'FULL_ADDER' && (
                                  <button 
                                    disabled={params.loopActive}
                                    onClick={() => setParams((p: any) => ({...p, logicState: [p.logicState[0], p.logicState[1], !p.logicState[2]]}))}
                                    className={`p-3 text-xs font-bold border transition-all ${params.logicState[2] ? 'bg-green-500 text-black border-green-400' : 'bg-black text-gray-500 border-gray-700'} col-span-2 ${params.loopActive ? 'opacity-50' : ''}`}
                                 >
                                     CARRY IN: {params.logicState[2] ? "1" : "0"}
                                 </button>
                             )}
                        </div>

                        {/* Logic Loop Button */}
                        <div className="mt-3 border-t border-gray-800 pt-3">
                             <button 
                                onClick={() => setParams((p: any) => ({...p, loopActive: !p.loopActive, logicState: [false,false,false]}))}
                                className={`w-full py-2 text-xs font-bold border transition-all ${params.loopActive ? 'bg-cyan-900/40 border-cyan-500 text-cyan-200 animate-pulse' : 'bg-gray-900 border-gray-600 text-gray-400 hover:border-gray-400'}`}
                             >
                                {params.loopActive ? "STOP SIMULATION LOOP" : "RUN SIMULATION LOOP"}
                             </button>
                        </div>
                        </>
                    )}

                    <TruthTable params={params} />

                    <div className="text-[10px] text-gray-400 font-mono text-center pt-2 h-4">
                        {params.gateType === 'XOR' && params.logicState[0] && params.logicState[1] ? "PHYSICS: DESTRUCTIVE INTERFERENCE" : ""}
                        {params.gateType === 'AND' && params.logicState[0] && params.logicState[1] ? "PHYSICS: CONSTRUCTIVE WAVE SUM" : ""}
                        {params.circuitMode === 'REGISTER_BANK' && params.registerState > 0 ? "PHYSICS: UNDAMPED PLASMA TOROID (LATCH)" : ""}
                    </div>
                </div>
            )
        }
        if (mode === 'temporal') {
             return (
                <div className="mt-4 space-y-2">
                    <div className="text-xs text-green-500 font-bold tracking-widest border-b border-green-900 pb-1">CAUSAL LEARNING</div>
                    <button 
                        onClick={() => {
                            setParams((p: any) => ({...p, inputText: "TICK", targetRegion: 0}));
                            setTimeout(() => setParams((p: any) => ({...p, inputText: "TOCK", targetRegion: 1})), 1000);
                            setTimeout(() => setParams((p: any) => ({...p, inputText: ""})), 2000);
                        }}
                        className="w-full py-2 bg-green-900/20 border border-green-500/50 hover:bg-green-500/20 text-green-400 text-xs font-bold"
                    >
                        RUN TRAINING CYCLE
                    </button>
                    <button 
                        onClick={() => {
                             setParams((p: any) => ({...p, inputText: "TICK", targetRegion: 0}));
                             setTimeout(() => setParams((p: any) => ({...p, inputText: ""})), 500);
                        }}
                        className="w-full py-2 bg-white/10 border border-white/20 hover:bg-white/20 text-white text-xs font-bold"
                    >
                        TRIGGER "TICK" (TEST)
                    </button>
                </div>
            )
        }
        return null;
    }

    return (
        <div className="absolute top-0 right-0 h-full w-80 p-4 pointer-events-none flex flex-col items-end">
            <div className="pointer-events-auto bg-black/80 backdrop-blur-md border border-cyan-900/50 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                <div className="flex justify-between items-center mb-4 border-b border-gray-800 pb-2">
                    <h2 className="text-cyan-400 font-bold tracking-widest text-sm">CONTROL UNIT</h2>
                    <div className="flex gap-2">
                        <button onClick={onShowInfo} className="w-6 h-6 flex items-center justify-center border border-cyan-500 text-cyan-500 hover:bg-cyan-500 hover:text-black text-xs font-bold">?</button>
                        <button onClick={onExit} className="w-6 h-6 flex items-center justify-center border border-red-500 text-red-500 hover:bg-red-500 hover:text-black text-xs font-bold">X</button>
                    </div>
                </div>
                
                {/* Global Params */}
                <div className="space-y-3 mb-2">
                     <div className="flex justify-between items-center">
                        <span className="text-[10px] text-gray-500 uppercase">Input Pattern</span>
                        <input 
                            type="text" 
                            value={params.inputText} 
                            onChange={(e) => setParams({...params, inputText: e.target.value.toUpperCase()})}
                            className="bg-black border border-gray-700 text-right text-xs text-white p-1 w-24 focus:border-cyan-500 outline-none"
                            disabled={mode === 'temporal' || mode === 'logic' || mode === 'adder'}
                        />
                     </div>
                     <div className="flex justify-between items-center">
                        <span className="text-[10px] text-gray-500 uppercase">Vis. Regions</span>
                        <button 
                            onClick={() => setParams({...params, showRegions: !params.showRegions})}
                            className={`text-[9px] px-2 py-1 border ${params.showRegions ? 'border-cyan-500 text-cyan-400' : 'border-gray-800 text-gray-600'}`}
                        >
                            {params.showRegions ? "ON" : "OFF"}
                        </button>
                     </div>
                </div>

                {renderControls()}
            </div>
            
            <div className="mt-auto pointer-events-auto bg-black/80 backdrop-blur-md border-t border-cyan-900 p-2 w-full text-[10px] font-mono text-gray-500">
                MODE: <span className="text-cyan-400">{mode.toUpperCase()}</span><br/>
                PARTICLES: <span className="text-white">{params.particleCount}</span><br/>
                PHYSICS: <span className={params.usePaperPhysics ? "text-purple-400" : "text-white"}>{params.usePaperPhysics ? "L-GROUP (SPIN)" : "STANDARD (NEWTON)"}</span>
            </div>
        </div>
    )
}

const InfoModal: React.FC<{ mode: string; onClose: () => void }> = ({ mode, onClose }) => {
    const info = EXPERIMENT_INFO[mode] || EXPERIMENT_INFO['standard'];
    if (!info) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
             <div className="max-w-2xl w-full bg-black border border-cyan-500 p-8 shadow-[0_0_50px_rgba(6,182,212,0.3)] relative font-['Rajdhani']">
                <button onClick={onClose} className="absolute top-4 right-4 text-cyan-500 hover:text-white font-bold text-xl">X</button>
                <h2 className="text-3xl font-bold text-cyan-400 mb-2 uppercase tracking-widest">{info.title}</h2>
                <div className="h-1 w-20 bg-cyan-600 mb-6"></div>
                
                <div className="space-y-6 text-gray-300 font-sans text-sm">
                    <div><h3 className="text-cyan-200 font-bold uppercase text-xs mb-1">Summary</h3><p>{info.summary}</p></div>
                    <div><h3 className="text-cyan-200 font-bold uppercase text-xs mb-1">Why It Matters</h3><p>{info.importance}</p></div>
                    <div><h3 className="text-cyan-200 font-bold uppercase text-xs mb-1">What to Do</h3><p>{info.demonstration}</p></div>
                    <div>
                        <h3 className="text-cyan-200 font-bold uppercase text-xs mb-1">Steps</h3>
                        <ul className="list-disc pl-5 space-y-1">{info.steps.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                </div>
             </div>
        </div>
    )
}

const RegionGuides: React.FC<{ params: SimulationParams }> = ({ params }) => {
  if (!params.showRegions) return null;

  if (params.logicMode) {
    return <LogicGateOverlay gateType={params.gateType} circuitMode={params.circuitMode} />;
  }

  return (
    <group>
      {/* Region 0: Input A / Cause */}
      <mesh position={[-35, 0, 0]}>
        <boxGeometry args={[25, 30, 25]} />
        <meshBasicMaterial color="#00FFFF" wireframe transparent opacity={0.1} />
      </mesh>
      <Text position={[-35, 18, 0]} fontSize={2} color="#00FFFF" anchorX="center" anchorY="middle">
        REGION A
      </Text>

      {/* Region 1: Input B / Effect */}
      <mesh position={[35, 0, 0]}>
        <boxGeometry args={[25, 30, 25]} />
        <meshBasicMaterial color="#FF00FF" wireframe transparent opacity={0.1} />
      </mesh>
      <Text position={[35, 18, 0]} fontSize={2} color="#FF00FF" anchorX="center" anchorY="middle">
        REGION B
      </Text>

      {/* Region 2: Processing / Output */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[30, 30, 30]} />
        <meshBasicMaterial color="#FFFFFF" wireframe transparent opacity={0.05} />
      </mesh>
    </group>
  );
};

const TitleScreen: React.FC<{ onStart: (mode: string) => void }> = ({ onStart }) => {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black text-white z-50 pointer-events-auto">
      <div className="max-w-4xl w-full p-8 relative">
        <div className="absolute top-0 left-0 w-32 h-32 border-t-2 border-l-2 border-cyan-500/50"></div>
        <div className="absolute bottom-0 right-0 w-32 h-32 border-b-2 border-r-2 border-cyan-500/50"></div>

        <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-2 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-white to-purple-400">
          MORPHIC
        </h1>
        <p className="text-xl md:text-2xl text-gray-400 font-light tracking-widest mb-12 uppercase">
          Thermodynamic Computing Engine
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <button onClick={() => onStart('standard')} className="group relative p-6 border border-gray-800 hover:border-cyan-500 transition-all bg-gray-900/50 hover:bg-cyan-900/20 text-left">
            <div className="text-cyan-500 text-xs font-bold tracking-widest mb-2">MODULE 01</div>
            <div className="text-2xl font-bold text-white group-hover:text-cyan-400 mb-2">GENESIS</div>
            <p className="text-xs text-gray-500 leading-relaxed">Standard predictive coding dynamics. Chaos to Order.</p>
          </button>

          <button onClick={() => onStart('inference')} className="group relative p-6 border border-gray-800 hover:border-yellow-500 transition-all bg-gray-900/50 hover:bg-yellow-900/20 text-left">
            <div className="text-yellow-500 text-xs font-bold tracking-widest mb-2">MODULE 02</div>
            <div className="text-2xl font-bold text-white group-hover:text-yellow-400 mb-2">AGENCY</div>
            <p className="text-xs text-gray-500 leading-relaxed">Active Inference and thermodynamic temperature control.</p>
          </button>

          <button onClick={() => onStart('temporal')} className="group relative p-6 border border-gray-800 hover:border-green-500 transition-all bg-gray-900/50 hover:bg-green-900/20 text-left">
            <div className="text-green-500 text-xs font-bold tracking-widest mb-2">MODULE 03</div>
            <div className="text-2xl font-bold text-white group-hover:text-green-400 mb-2">CAUSALITY</div>
            <p className="text-xs text-gray-500 leading-relaxed">Hebbian temporal learning and prediction.</p>
          </button>

          <button onClick={() => onStart('paper')} className="group relative p-6 border border-gray-800 hover:border-purple-500 transition-all bg-gray-900/50 hover:bg-purple-900/20 text-left">
            <div className="text-purple-500 text-xs font-bold tracking-widest mb-2">MODULE 04</div>
            <div className="text-2xl font-bold text-white group-hover:text-purple-400 mb-2">SPIN GROUP</div>
            <p className="text-xs text-gray-500 leading-relaxed">Lie Group dynamics with intrinsic spin states.</p>
          </button>

          <button onClick={() => onStart('logic')} className="group relative p-6 border border-gray-800 hover:border-red-500 transition-all bg-gray-900/50 hover:bg-red-900/20 text-left">
            <div className="text-red-500 text-xs font-bold tracking-widest mb-2">MODULE 05</div>
            <div className="text-2xl font-bold text-white group-hover:text-red-400 mb-2">LOGIC GATE</div>
            <p className="text-xs text-gray-500 leading-relaxed">XOR computation via wave interference.</p>
          </button>

          <button onClick={() => onStart('adder')} className="group relative p-6 border border-gray-800 hover:border-orange-500 transition-all bg-gray-900/50 hover:bg-orange-900/20 text-left">
            <div className="text-orange-500 text-xs font-bold tracking-widest mb-2">MODULE 06</div>
            <div className="text-2xl font-bold text-white group-hover:text-orange-400 mb-2">CPU CORE</div>
            <p className="text-xs text-gray-500 leading-relaxed">Complex circuits: Adders and Registers.</p>
          </button>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
    const [mode, setMode] = useState<string | null>(null);
    const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
    const [teacherFeedback, setTeacherFeedback] = useState(0);
    const [showInfo, setShowInfo] = useState(false);
    
    // Test State
    const [testResults, setTestResults] = useState<TestResult[]>([]);
    const [isTesting, setIsTesting] = useState(false);
  
    const dataRef = useRef<ParticleData>({
      x: new Float32Array(0), v: new Float32Array(0), phase: new Float32Array(0), spin: new Int8Array(0),
      activation: new Float32Array(0), target: new Float32Array(0), hasTarget: new Uint8Array(0),
      memoryMatrix: new Float32Array(0), regionID: new Uint8Array(0), forwardMatrix: new Float32Array(0),
      feedbackMatrix: new Float32Array(0), delayedActivation: new Float32Array(0), lastActiveTime: new Float32Array(0),
      hysteresisState: new Uint8Array(0),
    });
  
    const statsRef = useRef<SystemStats>({ meanError: 0, meanSpeed: 0, energy: 0, fps: 0, temperature: 0, isStable: false, trainingProgress: 0, phaseOrder: 0, spinOrder: 0, entropy: 0, patternMatch: 0 });
    const spatialRefs = useRef({ neighborList: new Int32Array(0), neighborCounts: new Int32Array(0), gridHead: new Int32Array(4096).fill(-1), gridNext: new Int32Array(0), frameCounter: 0 });
  
    // --- EXPERIMENT E LOOP LOGIC ---
    useEffect(() => {
        if (!params.loopActive || params.gateType !== 'HALF_ADDER') return;

        let cycle = 0;
        let intervalTime = 1500;
        
        const loopFunction = () => {
            // Complex Circuit Automation
            if (params.circuitMode === 'REGISTER_BANK') {
                 // Counts up 0-15 and stores it
                 setParams(current => ({ ...current, registerState: (current.registerState + 1) % 16 }));
            }
            else if (params.circuitMode === 'RIPPLE_ADDER') {
                 // Random additions
                 setParams(current => {
                     // If result odd, save? (Simulating conditional request in prompt)
                     const a = Math.floor(Math.random()*8);
                     const b = Math.floor(Math.random()*8);
                     return { ...current, inputA_4bit: a, inputB_4bit: b };
                 });
            } 
            else if (params.circuitMode === 'FULL_ADDER') {
                 // 8 states
                 setParams(current => {
                     const a = (cycle & 4) !== 0;
                     const b = (cycle & 2) !== 0;
                     const c = (cycle & 1) !== 0;
                     return { ...current, logicState: [a, b, c] };
                 });
                 cycle = (cycle + 1) % 8;
            }
            else {
                 // Half Adder
                 setParams(current => {
                     const a = (cycle & 2) !== 0;
                     const b = (cycle & 1) !== 0;
                     return { ...current, logicState: [a, b, false] };
                 });
                 cycle = (cycle + 1) % 4;
            }
        };

        const interval = setInterval(loopFunction, intervalTime);
        loopFunction(); // Run immediate

        return () => clearInterval(interval);
    }, [params.loopActive, params.gateType, params.circuitMode]);


    const handleStart = (selectedMode: string) => {
        setMode(selectedMode);
        
        const newParams = { ...DEFAULT_PARAMS };
        
        if (selectedMode === 'inference') {
            newParams.inputText = "ORDER";
            newParams.targetRegion = 2; // Center
        } else if (selectedMode === 'temporal') {
            newParams.inputText = "";
            newParams.plasticity = 0.1;
        } else if (selectedMode === 'paper') {
            newParams.usePaperPhysics = true;
            newParams.inputText = "L-GROUP";
            newParams.particleCount = 1600;
        } else if (selectedMode === 'logic') {
            newParams.logicMode = true;
            newParams.usePaperPhysics = true;
            newParams.inputText = "LOGIC";
            newParams.particleCount = 300; 
        } else if (selectedMode === 'adder') {
            newParams.logicMode = true;
            newParams.usePaperPhysics = true;
            newParams.gateType = 'HALF_ADDER';
            newParams.inputText = "ADDER";
            newParams.particleCount = 600; // Increased for 4-bit complexity
            newParams.accumulator = 0;
            newParams.loopActive = false;
            newParams.circuitMode = 'HALF_ADDER';
            newParams.memoryBit = 0;
        }

        setParams(newParams);
    };

    const handleExit = () => {
        setMode(null);
        setParams(DEFAULT_PARAMS);
    };

    const runLogicTests = async () => {
        if (isTesting) return;
        setIsTesting(true);
        setTestResults([]);
        
        setParams(p => ({ ...p, logicState: [false, false, false] }));
        await new Promise(resolve => setTimeout(resolve, 500));
    
        const count = params.particleCount;
        const currentGate = params.gateType;

        const getExpected = (a: boolean, b: boolean) => {
             if (currentGate === 'HALF_ADDER') return (a && b) ? 1.0 : 0.0; // Test Carry

            switch(currentGate) {
                case 'AND': return (a && b) ? 1.0 : 0.0;
                case 'OR': return (a || b) ? 1.0 : 0.0;
                case 'XOR': return (a !== b) ? 1.0 : 0.0;
                case 'NAND': return !(a && b) ? 1.0 : 0.0;
                case 'NOR': return !(a || b) ? 1.0 : 0.0;
                case 'XNOR': return (a === b) ? 1.0 : 0.0;
                case 'NOT': return (!a) ? 1.0 : 0.0;
                default: return 0.0;
            }
        }
        
        let cases = [
            { a: false, b: false, label: "INPUT: 00" },
            { a: true, b: false, label: "INPUT: 10" },
            { a: false, b: true, label: "INPUT: 01" },
            { a: true, b: true, label: "INPUT: 11" }
        ];

        if (currentGate === 'NOT') {
             cases = [
                 { a: false, b: false, label: "INPUT: 0" },
                 { a: true, b: false, label: "INPUT: 1" }
             ];
        }
    
        const newResults: TestResult[] = [];
    
        for (const c of cases) {
            dataRef.current.v.fill(0);
            dataRef.current.activation.fill(0);

            const expectedVal = getExpected(c.a, c.b);

            setParams(p => ({ ...p, logicState: [c.a, c.b, false] }));
            
            await new Promise(resolve => setTimeout(resolve, 1200));
    
            let totalAct = 0;
            let pCount = 0;
            
            let targetRegion = 2;
            if (currentGate === 'HALF_ADDER') targetRegion = 3; 
            
            for (let i = 0; i < count; i++) {
                 if (dataRef.current.regionID[i] === targetRegion) {
                     totalAct += dataRef.current.activation[i];
                     pCount++;
                 }
            }
            
            const avg = pCount > 0 ? totalAct / pCount : 0;
            
            let passed = false;
            if (expectedVal > 0.5) {
                passed = avg > 0.6;
            } else {
                passed = avg < 0.2; 
            }
    
            newResults.push({
                testName: c.label,
                score: avg,
                maxScore: 1.0,
                status: passed ? 'PASS' : 'FAIL',
                details: `Activ: ${avg.toFixed(2)}`
            });
            
            setTestResults([...newResults]);
        }
    
        setIsTesting(false);
        setParams(p => ({ ...p, logicState: [false, false, false] }));
    };

    return (
      <div style={{ width: '100vw', height: '100vh', background: '#000', overflow: 'hidden' }}>
        
        {/* SIMULATION LAYER */}
        {mode && (
            <Canvas camera={{ position: [0, 0, 70], fov: 45 }} gl={{ antialias: false }}>
                <color attach="background" args={['#050505']} />
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 10, 5]} intensity={1.5} />
                <pointLight position={[30, 30, 30]} intensity={1} color="#ffaa00" />
                <pointLight position={[-30, -30, 30]} intensity={0.5} color="#00aaff" />
                
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                <Grid position={[0, -20, 0]} args={[100, 100]} cellSize={4} sectionSize={20} sectionColor="#06b6d4" cellColor="#1e293b" fadeDistance={60} />

                <ParticleSystem 
                    params={params} 
                    dataRef={dataRef} 
                    statsRef={statsRef} 
                    started={true} 
                    spatialRefs={spatialRefs}
                    teacherFeedback={teacherFeedback}
                />
                
                <RegionGuides params={params} />

                <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                
                <EffectComposer enableNormalPass={false}>
                    <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} intensity={1.5} />
                    <Noise opacity={0.05} />
                    <Vignette eskil={false} offset={0.1} darkness={1.1} />
                </EffectComposer>
            </Canvas>
        )}

        {/* UI LAYER */}
        {!mode && <TitleScreen onStart={handleStart} />}
        
        {mode && (
            <UIOverlay 
                mode={mode} 
                params={params} 
                setParams={setParams} 
                setFeedback={setTeacherFeedback} 
                onExit={handleExit}
                onShowInfo={() => setShowInfo(true)}
                testResults={testResults} 
                isTesting={isTesting}    
                onRunTests={runLogicTests} 
            />
        )}

        {showInfo && mode && <InfoModal mode={mode} onClose={() => setShowInfo(false)} />}

      </div>
    );
  };
  
  export default App;