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
    
    // --- SPECIAL LOGIC MODE LAYOUT (Y-Junction) ---
    if (params.logicMode) {
        const count = params.particleCount;
        data.current.hasTarget.fill(1);
        
        // Split particles into 3 defined channel groups
        const channelSize = Math.floor(count / 3);
        
        for(let i=0; i<count; i++) {
            let cx=0, cy=0, cz=0;
            let group = 0; // 0=InputA, 1=InputB, 2=Output

            if (i < channelSize) group = 0; 
            else if (i < channelSize * 2) group = 1; 
            else group = 2;

            data.current.regionID[i] = group; // Force region ID to match channel

            // Linear interpolation variable (0 to 1 along the path)
            const t = (i % channelSize) / channelSize;
            
            // Y-Junction Coordinates
            // Input A: Top Left (-35, 15) -> Center (-5, 0)
            // Input B: Bottom Left (-35, -15) -> Center (-5, 0)
            // Output: Center (-5, 0) -> Right (35, 0)

            const jitter = 1.0; // Reduced jitter for cleaner lines

            if (group === 0) {
                 // Path A
                 cx = -35 * (1-t) + (-5) * t;
                 cy = 15 * (1-t) + (0) * t;
                 cz = (Math.random() - 0.5) * jitter;
                 cy += (Math.random() - 0.5) * jitter;
            } else if (group === 1) {
                 // Path B
                 cx = -35 * (1-t) + (-5) * t;
                 cy = -15 * (1-t) + (0) * t;
                 cz = (Math.random() - 0.5) * jitter;
                 cy += (Math.random() - 0.5) * jitter;
            } else {
                 // Output
                 cx = -5 * (1-t) + 35 * t;
                 cy = (Math.random() - 0.5) * jitter;
                 cz = (Math.random() - 0.5) * jitter;
            }

            data.current.target[i*3] = cx;
            data.current.target[i*3+1] = cy;
            data.current.target[i*3+2] = cz;
            
            data.current.activation[i] = 0.0;
            data.current.spin[i] = 1; // Unify spin
        }
        return;
    }

    if (!params.inputText) {
        data.current.hasTarget.fill(0);
        return;
    }

    const count = params.particleCount;
    // Lower density slightly for text to make it cleaner in regions
    const { positions, count: pointCount } = textToPoints(params.inputText, Math.floor(count * 0.8));
    
    data.current.hasTarget.fill(0);

    // Calculate Offset based on Region
    // Region A (0): x < 0 | Region B (1): x > 0
    let offsetX = 0;
    if (params.targetRegion === 0) offsetX = -35;
    else if (params.targetRegion === 1) offsetX = 35;

    if (pointCount > 0) {
        const targets = [];
        for(let i=0; i<pointCount; i++) {
            targets.push({ 
                x: positions[i*3] + offsetX, 
                y: positions[i*3+1], 
                z: positions[i*3+2] 
            });
        }
        
        // Shuffle targets
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
            
            // Region Filter
            if (params.targetRegion !== -1 && data.current.regionID[pid] !== params.targetRegion) {
                continue;
            }

            const t = targets[i];
            data.current.target[pid * 3] = t.x;
            data.current.target[pid * 3 + 1] = t.y;
            data.current.target[pid * 3 + 2] = t.z;
            data.current.hasTarget[pid] = 1;
            data.current.activation[pid] = 1.0;
        }
    }
  }, [params.inputText, params.particleCount, params.targetRegion, params.paused, params.logicMode]);

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

    const effectiveChaos = started ? params.chaosMode : false;
    const timeNow = state.clock.elapsedTime;
    
    // Teacher Algorithm: Thermodynamic Modulation
    let systemTemperature = 0.0;
    if (teacherFeedback === 1) systemTemperature = 0.0; // Crystallize
    else if (teacherFeedback === -1) systemTemperature = 1.0; // Agitate
    else systemTemperature = 0.05; // Ambient

    const { equilibriumDistance, stiffness, plasticity, phaseSyncRate, usePaperPhysics, spinCouplingStrength, phaseCouplingStrength, logicMode, logicState } = params;
    const count = params.particleCount;
    const { x, v, phase, spin, target, hasTarget, regionID, forwardMatrix, activation, delayedActivation, lastActiveTime, hysteresisState } = data.current;
    
    if (x.length === 0) return;

    const effectivePlasticity = teacherFeedback === -1 ? 0.3 : (teacherFeedback === 1 ? 0.0 : plasticity);
    const isEncoding = effectivePlasticity > 0;
    const isRecalling = params.inputText === "" && !logicMode; 

    const k_spring_base = isEncoding ? 0.2 : (isRecalling ? 1.5 : (started ? 0.2 : 0.05));
    const stiffnessMult = isRecalling ? 0.1 : (isEncoding ? 0.1 : 1.0);

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
                                 
                                 // Logic Mode: Use tighter interaction for clean visualization
                                 const neighborDistLimit = logicMode ? 10.0 : 25.0; 
                                 
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

    let lineIndex = 0;
    const linePositions = linesRef.current.geometry.attributes.position.array as Float32Array;
    const lineColors = linesRef.current.geometry.attributes.color.array as Float32Array;
    let totalError = 0;
    let totalSpeed = 0;
    
    // Stats accumulators
    let sumCosPhase = 0;
    let sumSinPhase = 0;
    let netSpin = 0;

    // CONTINUOUS STDP PARAMETERS
    const continuousStdpRate = 0.05; 
    const delayAlpha = 0.1;

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0; 
      const idx3 = i * 3;
      const ix = x[idx3], iy = x[idx3 + 1], iz = x[idx3 + 2];
      const rid = regionID[i];
      const nOffset = i * 48; 
      const nCount = spatialRefs.current.neighborCounts[i];

      // --- LOGIC GATE SIMULATION OVERRIDE ---
      if (logicMode) {
          // Channel flow visualization logic
          const channelSize = Math.floor(count / 3);
          const t = (i % channelSize) / channelSize; // Progress along channel (0 to 1)
          
          let isActive = false;
          let flowPhase = 0;

          // Interference Calculation for Output Channel
          const inputA = logicState[0] ? 1.0 : 0.0;
          const inputB = logicState[1] ? -1.0 : 0.0; // Note negative phase for B
          const combined = inputA + inputB; 
          const isInterference = (inputA !== 0 && inputB !== 0);
          const outputActive = Math.abs(combined) > 0.5;

          if (rid === 0) { // Input A
              isActive = logicState[0];
              flowPhase = 1.0;
          } else if (rid === 1) { // Input B
              isActive = logicState[1];
              flowPhase = -1.0;
          } else if (rid === 2) { // Output
              isActive = outputActive;
              flowPhase = combined; 
          }

          if (isActive) {
              activation[i] = 1.0;
              // Distinct "Pulse" Packet Visualization
              // Create discrete packets moving along the wire
              const pulseFreq = 3.0; // Frequency of packets
              const pulseSpeed = 5.0; // Speed of packets
              const phaseOffset = t * 20.0 - timeNow * pulseSpeed;
              const pulse = Math.sin(phaseOffset);
              
              // Sharpen the pulse to look like a digital signal
              const sharpPulse = Math.pow((pulse + 1.0) * 0.5, 8.0); // High power makes peaks narrow

              // Apply Flow Color
              if (flowPhase > 0.5) TEMP_COLOR.copy(CYAN);
              else if (flowPhase < -0.5) TEMP_COLOR.copy(MAGENTA);
              else TEMP_COLOR.copy(DARK); 

              // Add wave pulse to color
              TEMP_COLOR.lerp(WHITE, sharpPulse);
              
              // Force scale based on pulse
              const scale = 0.2 + sharpPulse * 0.3;
              TEMP_OBJ.scale.set(scale, scale, scale);

          } else {
              activation[i] *= 0.9; // Fast decay
              
              // Visualization of Destructive Interference (Collision) at the Gate Input
              if (rid === 2 && isInterference) {
                 // Collision Zone (Start of output pipe) where t is small
                 if (t < 0.25) {
                     const jitterIntensity = 1.0 - (t / 0.25);
                     // Energetic chaotic motion
                     const jitter = (Math.random()-0.5) * 1.5 * jitterIntensity;
                     fx += jitter; 
                     fy += jitter; 
                     
                     // Color: Energy sparks (Orange/Red) fading to dark
                     // Simulates heat/entropy from cancellation
                     const spark = Math.random() > 0.7 ? 1.0 : 0.0;
                     TEMP_COLOR.setRGB(
                        (0.8 * jitterIntensity) + spark, 
                        (0.2 * jitterIntensity) + spark * 0.5, 
                        (0.1 * jitterIntensity)
                     );
                     
                     const scale = 0.15 * jitterIntensity + (spark * 0.15);
                     TEMP_OBJ.scale.set(scale, scale, scale);
                 } else {
                     // Completely blocked/cancelled downstream
                     TEMP_COLOR.setHex(0x050505); 
                     TEMP_OBJ.scale.set(0.05, 0.05, 0.05);
                 }
              } else {
                 TEMP_COLOR.setHex(0x111111); // Idle dim path
                 TEMP_OBJ.scale.set(0.1, 0.1, 0.1);
              }
          }
          
          if(meshRef.current) meshRef.current.setColorAt(i, TEMP_COLOR);
          
      } 
      // --- STANDARD PHYSICS LOGIC ---
      else {
          sumCosPhase += Math.cos(phase[i]);
          sumSinPhase += Math.sin(phase[i]);
          netSpin += spin[i];
      }

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
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          totalError += dist;
          
          // Logic Mode: Tighter spring to keep stream shape
          const k = logicMode ? 1.0 : k_spring_base;
          
          fx += dx * k;
          fy += dy * k;
          fz += dz * k;
      }

      if (!logicMode) {
          // ... (Standard interaction logic preserved for other modes) ...
          
          const baseInteractionStrength = hasTarget[i] ? 0.1 : 1.0;
          let predictionLock = 0.0;
          let phasorX = 0;
          let phasorY = 0;

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
                    
                    const proximity = Math.max(0, 1.0 - dist / 6.0);
                    const activationWeight = activation[j]; 
                    const weight = spinTerm * proximity * 0.2 * activationWeight; 
                    phasorX += weight * Math.cos(phase[j]);
                    phasorY += weight * Math.sin(phase[j]);
                }
                
                let springF = 0;
                if (dist < r0) springF = -stiffness * stiffnessMult * (r0 - dist) * 2.0;
                else if (!hasTarget[i]) springF = stiffness * stiffnessMult * (dist - r0) * 0.1;

                if (usePaperPhysics) {
                    force += springF * paperCoupling;
                } else {
                    force += springF;
                }

                if (!usePaperPhysics) {
                    const weightJI = forwardMatrix[j * count + i]; 
                    if (weightJI > 0.01) {
                        const signal = weightJI * delayedActivation[j];
                        predictionLock += signal; 
                        force += signal * 0.01; 
                    }
                }
                
                if (isEncoding && !usePaperPhysics) {
                    const hebbianProduct = activation[i] * delayedActivation[j];
                    if (hebbianProduct > 0.1) {
                        const deltaW = hebbianProduct * continuousStdpRate;
                        forwardMatrix[j * count + i] += deltaW;
                        if (forwardMatrix[j * count + i] > 1.0) forwardMatrix[j * count + i] = 1.0;
                    }
                }

                force *= baseInteractionStrength;
                const invDist = 1.0 / dist;
                fx += dx * invDist * force; fy += dy * invDist * force; fz += dz * invDist * force;
                
                if (usePaperPhysics) {
                     const syncRate = phaseSyncRate * (1.0 + spinCouplingStrength * spin[i] * spin[j]);
                     phaseDelta += Math.sin(phaseDiff) * 0.1 * syncRate;
                }

                const showLine = usePaperPhysics 
                    ? (paperCoupling > 0.8 && dist < r0 * 2.5) 
                    : (j > i && dist < r0 * 2.0);

                if (showLine && lineIndex < maxConnections) {
                    const li = lineIndex * 6;
                    linePositions[li] = ix; linePositions[li+1] = iy; linePositions[li+2] = iz;
                    linePositions[li+3] = x[j*3]; linePositions[li+4] = x[j*3+1]; linePositions[li+5] = x[j*3+2];
                    
                    if (usePaperPhysics) {
                        const hue = Math.abs(Math.cos(phaseDiff));
                        lineColors[li] = hue * 2; lineColors[li+1] = hue; lineColors[li+2] = 0.5;
                        lineColors[li+3] = hue * 2; lineColors[li+4] = hue; lineColors[li+5] = 0.5;
                    } else {
                        lineColors[li] = 0; lineColors[li+1] = 0.3; lineColors[li+2] = 0.8; 
                        lineColors[li+3] = 0; lineColors[li+4] = 0.3; lineColors[li+5] = 0.8;
                    }
                    lineIndex++;
                }
              }
          }
      } // End of non-logic physics

      let particleDamping = 0.85; 
      if (logicMode) particleDamping = 0.6; // Heavier damping for steady flow

      if (usePaperPhysics && !logicMode) {
          particleDamping = 0.90; 
      } 

      if (!started) particleDamping = 0.95; 
      else if (effectiveChaos) particleDamping = 0.98;
      
      v[idx3] = v[idx3] * particleDamping + fx;
      v[idx3+1] = v[idx3+1] * particleDamping + fy;
      v[idx3+2] = v[idx3+2] * particleDamping + fz;
      
      const speedSq = v[idx3]**2 + v[idx3+1]**2 + v[idx3+2]**2;
      const speed = Math.sqrt(speedSq);
      totalSpeed += speed;

      // UPDATE FLASH INTENSITY
      if (flashRef.current) {
          flashRef.current[i] *= 0.92;
          if (isRecalling && speed < 0.005) {
               flashRef.current[i] = 3.0; 
          }
      }

      x[idx3] += v[idx3]; x[idx3+1] += v[idx3+1]; x[idx3+2] += v[idx3+2];
      
      const rSq = x[idx3]**2 + x[idx3+1]**2 + x[idx3+2]**2;
      if (rSq > 2500) { x[idx3]*=0.99; x[idx3+1]*=0.99; x[idx3+2]*=0.99; }
      
      if (!logicMode) phase[i] += phaseSyncRate * phaseDelta + 0.05;

      // --- HYSTERESIS ---
      let externalDrive = hasTarget[i] ? 1.0 : 0.0;

      // Logic Mode skips complex hysteresis / activation dynamics in favor of explicit flow control above
      if (!logicMode) {
          const totalInputEnergy = externalDrive; // Simplified for standard
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

          if (activation[i] > 0.8 && (timeNow - lastActiveTime[i] > 0.5)) {
              lastActiveTime[i] = timeNow;
          }

          // Update Visuals for Non-Logic Modes
          TEMP_OBJ.position.set(x[idx3], x[idx3+1], x[idx3+2]);
          const s = hasTarget[i] ? 0.25 : 0.35; 
          TEMP_OBJ.scale.set(s, s, s);
      } else {
          // Logic Mode Update Matrix Position only (Scale/Color handled in override block)
           TEMP_OBJ.position.set(x[idx3], x[idx3+1], x[idx3+2]);
      }

      TEMP_OBJ.updateMatrix();
      if(meshRef.current) meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
      if(outlineRef.current) outlineRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      if (!logicMode) {
          // COLORING LOGIC for Standard Modes
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

          const pulse = 1.0 + Math.sin(phase[i]) * 0.3;
          const glowIntensity = usePaperPhysics ? 3.5 : 2.5 + entropy * 4.0; 
          TEMP_EMISSIVE.setRGB(r * glowIntensity * pulse, g * glowIntensity * pulse, b * glowIntensity * pulse);
          if(outlineRef.current) outlineRef.current.setColorAt(i, TEMP_EMISSIVE);
      } else {
          // Logic Mode Glow (Color already set in override block)
          TEMP_EMISSIVE.copy(TEMP_COLOR).multiplyScalar(2.0);
          if(outlineRef.current) outlineRef.current.setColorAt(i, TEMP_EMISSIVE);
      }

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

const LogicGateOverlay: React.FC = () => {
  const points = useMemo(() => {
     // XOR Gate Shape coordinates
     // Center approx (-5, 0)
     const shape = [];
     
     // Back Curve (concave)
     const backCurve = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-10, 8, 0),
         new THREE.Vector3(-7, 0, 0),
         new THREE.Vector3(-10, -8, 0)
     ]);
     shape.push(backCurve.getPoints(20));

     // Top Curve
     const topCurve = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-10, 8, 0),
         new THREE.Vector3(0, 8, 0), // Control
         new THREE.Vector3(5, 0, 0)  // Tip
     ]);
     shape.push(topCurve.getPoints(20));

      // Bottom Curve
     const bottomCurve = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-10, -8, 0),
         new THREE.Vector3(0, -8, 0), // Control
         new THREE.Vector3(5, 0, 0)  // Tip
     ]);
     shape.push(bottomCurve.getPoints(20));
     
     // Input Lines Visualization (Thin guides)
     shape.push([new THREE.Vector3(-35, 15, 0), new THREE.Vector3(-9, 4, 0)]);
     shape.push([new THREE.Vector3(-35, -15, 0), new THREE.Vector3(-9, -4, 0)]);
     
     // Output Line Visualization
     shape.push([new THREE.Vector3(5, 0, 0), new THREE.Vector3(35, 0, 0)]);

     // Inner Back Curve (XOR specific second line)
     const innerBack = new THREE.CatmullRomCurve3([
         new THREE.Vector3(-12, 8, 0),
         new THREE.Vector3(-9, 0, 0),
         new THREE.Vector3(-12, -8, 0)
     ]);
     shape.push(innerBack.getPoints(20));

     return shape;
  }, []);

  return (
      <group position={[-5, 0, 0]}>
          {points.map((p, i) => (
              <Line key={i} points={p} color="cyan" opacity={0.2} transparent lineWidth={1} />
          ))}
          <Text position={[-30, 18, 0]} fontSize={2} color="#00FFFF" anchorX="center" anchorY="middle">INPUT A</Text>
          <Text position={[-30, -18, 0]} fontSize={2} color="#FF00FF" anchorX="center" anchorY="middle">INPUT B</Text>
          <Text position={[40, 0, 0]} fontSize={2} color="white" anchorX="center" anchorY="middle">OUTPUT</Text>
          <Text position={[-2, 12, 0]} fontSize={1.5} color="gray" anchorX="center" anchorY="middle">XOR GATE</Text>
      </group>
  )
}

const RegionGuides: React.FC<{ params: SimulationParams }> = ({ params }) => {
    if (!params.showRegions && !params.logicMode) return null;
    if (params.logicMode) return <LogicGateOverlay />;
    
    return (
        <group>
            <Cylinder args={[0.5, 0.5, 40, 8]} position={[-35, 0, 0]} rotation={[0, 0, 0]}>
                <meshBasicMaterial color="#ff0055" transparent opacity={0.2} />
            </Cylinder>
             <Cylinder args={[0.5, 0.5, 40, 8]} position={[35, 0, 0]} rotation={[0, 0, 0]}>
                <meshBasicMaterial color="#0055ff" transparent opacity={0.2} />
            </Cylinder>
        </group>
    );
};

// --- UI Components ---

const TruthTable: React.FC<{ a: boolean, b: boolean }> = ({ a, b }) => {
    const rowClass = (ra: boolean, rb: boolean) => 
        (a === ra && b === rb) ? "bg-cyan-900/50 text-white font-bold border border-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.3)]" : "text-gray-600";
    
    return (
        <div className="mt-4 p-2 bg-black/40 border border-gray-800 backdrop-blur-sm">
            <div className="text-[10px] text-gray-500 font-mono mb-2 uppercase tracking-widest text-center border-b border-gray-800 pb-1">Logic Truth Table (XOR)</div>
            <div className="grid grid-cols-3 gap-1 text-xs font-mono text-center">
                <div className="text-gray-400 font-bold pb-1">A</div>
                <div className="text-gray-400 font-bold pb-1">B</div>
                <div className="text-gray-400 font-bold pb-1">OUT</div>

                <div className={`p-1 rounded transition-colors ${rowClass(false, false)}`}>0</div>
                <div className={`p-1 rounded transition-colors ${rowClass(false, false)}`}>0</div>
                <div className={`p-1 rounded transition-colors ${rowClass(false, false)}`}>0</div>

                <div className={`p-1 rounded transition-colors ${rowClass(false, true)}`}>0</div>
                <div className={`p-1 rounded transition-colors ${rowClass(false, true)}`}>1</div>
                <div className={`p-1 rounded transition-colors ${rowClass(false, true)}`}>1</div>

                <div className={`p-1 rounded transition-colors ${rowClass(true, false)}`}>1</div>
                <div className={`p-1 rounded transition-colors ${rowClass(true, false)}`}>0</div>
                <div className={`p-1 rounded transition-colors ${rowClass(true, false)}`}>1</div>

                <div className={`p-1 rounded transition-colors ${rowClass(true, true)}`}>1</div>
                <div className={`p-1 rounded transition-colors ${rowClass(true, true)}`}>1</div>
                <div className={`p-1 rounded transition-colors ${rowClass(true, true)}`}>0</div>
            </div>
        </div>
    )
}

const TitleScreen: React.FC<{ onStart: (mode: string) => void }> = ({ onStart }) => {
    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/90 text-white font-['Rajdhani']">
            <div className="flex flex-col items-start space-y-6 max-w-4xl w-full p-10 border-l-4 border-cyan-500 bg-black/50 backdrop-blur-md">
                <div className="text-sm text-cyan-500 tracking-[0.3em] opacity-80 mb-[-10px]">QUANTUM PREDICTIVE CODING LAB</div>
                <h1 className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-white to-purple-500">
                    PREDICTIVE MORPHOLOGY
                </h1>
                <p className="max-w-xl text-gray-400 font-mono text-sm leading-relaxed border-t border-gray-800 pt-4">
                    Explore the thermodynamics of active inference. Simulate biological memory, causal prediction, and quantum logic gates using a particle-based holographic neural network.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full mt-8">
                     <button onClick={() => onStart('standard')} className="text-left p-4 border border-gray-700 hover:border-cyan-400 hover:bg-cyan-900/20 transition-all group">
                        <div className="text-cyan-400 font-bold tracking-widest text-xs mb-1 group-hover:text-cyan-300">SANDBOX</div>
                        <div className="text-xl font-bold text-white">STANDARD SIMULATION</div>
                        <div className="text-xs text-gray-500 mt-2 font-mono">Baseline free energy minimization.</div>
                    </button>

                    <button onClick={() => onStart('inference')} className="text-left p-4 border border-gray-700 hover:border-yellow-400 hover:bg-yellow-900/20 transition-all group">
                        <div className="text-yellow-400 font-bold tracking-widest text-xs mb-1 group-hover:text-yellow-300">EXPERIMENT A</div>
                        <div className="text-xl font-bold text-white">ACTIVE INFERENCE</div>
                        <div className="text-xs text-gray-500 mt-2 font-mono">Thermodynamic agency & stabilization.</div>
                    </button>

                    <button onClick={() => onStart('temporal')} className="text-left p-4 border border-gray-700 hover:border-green-400 hover:bg-green-900/20 transition-all group">
                        <div className="text-green-400 font-bold tracking-widest text-xs mb-1 group-hover:text-green-300">EXPERIMENT B</div>
                        <div className="text-xl font-bold text-white">CAUSAL PREDICTION</div>
                        <div className="text-xs text-gray-500 mt-2 font-mono">Hebbian temporal learning (STDP).</div>
                    </button>
                    
                    <button onClick={() => onStart('paper')} className="text-left p-4 border border-gray-700 hover:border-purple-400 hover:bg-purple-900/20 transition-all group">
                        <div className="text-purple-400 font-bold tracking-widest text-xs mb-1 group-hover:text-purple-300">EXPERIMENT C</div>
                        <div className="text-xl font-bold text-white">L-GROUP DYNAMICS</div>
                        <div className="text-xs text-gray-500 mt-2 font-mono">Spin/Phase vibrational coupling.</div>
                    </button>

                    <button onClick={() => onStart('logic')} className="text-left p-4 border border-gray-700 hover:border-red-400 hover:bg-red-900/20 transition-all group">
                        <div className="text-red-400 font-bold tracking-widest text-xs mb-1 group-hover:text-red-300">EXPERIMENT D</div>
                        <div className="text-xl font-bold text-white">LOGIC GATE (XOR)</div>
                        <div className="text-xs text-gray-500 mt-2 font-mono">Destructive interference computation.</div>
                    </button>
                </div>
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
        if (mode === 'logic') {
            return (
                <div className="mt-4 space-y-2">
                    <div className="text-xs text-red-500 font-bold tracking-widest border-b border-red-900 pb-1">QUANTUM GATES (XOR)</div>
                    <div className="grid grid-cols-2 gap-2">
                         <button 
                            onClick={() => setParams((p: any) => ({...p, logicState: [!p.logicState[0], p.logicState[1]]}))}
                            className={`p-3 text-xs font-bold border transition-all ${params.logicState[0] ? 'bg-cyan-500 text-black border-cyan-400' : 'bg-black text-gray-500 border-gray-700'}`}
                         >
                             INPUT A: {params.logicState[0] ? "1" : "0"}
                         </button>
                         <button 
                            onClick={() => setParams((p: any) => ({...p, logicState: [p.logicState[0], !p.logicState[1]]}))}
                            className={`p-3 text-xs font-bold border transition-all ${params.logicState[1] ? 'bg-pink-500 text-black border-pink-400' : 'bg-black text-gray-500 border-gray-700'}`}
                         >
                             INPUT B: {params.logicState[1] ? "1" : "0"}
                         </button>
                    </div>

                    <TruthTable a={params.logicState[0]} b={params.logicState[1]} />

                    <div className="text-[10px] text-gray-400 font-mono text-center pt-2">
                        {(params.logicState[0] && params.logicState[1]) 
                            ? "PHASE CANCELLATION DETECTED" 
                            : (params.logicState[0] || params.logicState[1]) 
                                ? "SIGNAL FLOW ACTIVE" 
                                : "AWAITING SIGNAL"
                        }
                    </div>

                     {/* Test Runner UI */}
                    <div className="border-t border-gray-800 pt-3 mt-3">
                         <button 
                            onClick={onRunTests}
                            disabled={isTesting}
                            className={`w-full py-2 text-xs font-bold border ${isTesting ? 'border-gray-600 text-gray-500 cursor-not-allowed' : 'border-cyan-500 text-cyan-400 hover:bg-cyan-900/40'}`}
                         >
                            {isTesting ? "RUNNING DIAGNOSTICS..." : "RUN AUTOMATED TEST SUITE"}
                         </button>
                         
                         {testResults.length > 0 && (
                             <div className="mt-2 space-y-1">
                                 {testResults.map((res, i) => (
                                     <div key={i} className="flex justify-between items-center text-[10px] font-mono bg-black/40 p-1 px-2 border-l-2 border-gray-700">
                                         <span className="text-gray-400">{res.testName}</span>
                                         <div className="flex items-center gap-2">
                                             <span className="text-gray-500">{res.details}</span>
                                             <span className={`font-bold ${res.status === 'PASS' ? 'text-green-400' : 'text-red-500'}`}>{res.status}</span>
                                         </div>
                                     </div>
                                 ))}
                             </div>
                         )}
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
                            disabled={mode === 'temporal' || mode === 'logic'}
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
  
    const handleStart = (selectedMode: string) => {
        setMode(selectedMode);
        
        // Reset Params based on Mode
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
            newParams.inputText = "LOGIC"; // Placeholder, layout driven by logicMode
            newParams.particleCount = 320; // Reduced count for cleaner circuit flow
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
    
        // Capture count from current params to ensure consistency
        const count = params.particleCount;
        const channelSize = Math.floor(count / 3);
        
        const cases = [
            { a: false, b: false, expected: 0.0, label: "INPUT: 00" },
            { a: true, b: false, expected: 1.0, label: "INPUT: 10" },
            { a: false, b: true, expected: 1.0, label: "INPUT: 01" },
            { a: true, b: true, expected: 0.0, label: "INPUT: 11 (XOR)" }
        ];
    
        const newResults: TestResult[] = [];
    
        for (const c of cases) {
            // 1. Set Input
            setParams(p => ({ ...p, logicState: [c.a, c.b] }));
            
            // 2. Wait for signal propagation (simulated)
            await new Promise(resolve => setTimeout(resolve, 800));
    
            // 3. Measure Output Region (Region 2)
            let totalAct = 0;
            let pCount = 0;
            // Region 2 starts at index 2 * channelSize
            const startIdx = channelSize * 2;
            
            for (let i = startIdx; i < count; i++) {
                 totalAct += dataRef.current.activation[i];
                 pCount++;
            }
            
            const avg = pCount > 0 ? totalAct / pCount : 0;
            // Threshold for pass: Expected 1.0 (allow >0.8), Expected 0.0 (allow <0.2)
            const passed = Math.abs(avg - c.expected) < 0.25;
    
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
        // Reset inputs
        setParams(p => ({ ...p, logicState: [false, false] }));
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