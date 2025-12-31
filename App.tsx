import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { SimulationParams, ParticleData, DEFAULT_PARAMS, MemoryAction, CONSTANTS } from './types';

// --- ParticleSystem ---
// Workaround for missing JSX types in current environment
const InstancedMesh = 'instancedMesh' as any;
const SphereGeometry = 'sphereGeometry' as any;
const MeshStandardMaterial = 'meshStandardMaterial' as any;
const MeshBasicMaterial = 'meshBasicMaterial' as any;
const LineSegments = 'lineSegments' as any;
const LineBasicMaterial = 'lineBasicMaterial' as any;

const TEMP_OBJ = new THREE.Object3D();
const TEMP_COLOR = new THREE.Color();

const textToPoints = (text: string): { positions: Float32Array, count: number } => {
  if (!text) return { positions: new Float32Array(0), count: 0 };

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { positions: new Float32Array(0), count: 0 };

  const L = text.length;

  // Robust Inverse-Power Scaling
  const maxFontSize = 340;
  const minFontSize = 50;
  
  const fontSize = Math.max(minFontSize, maxFontSize / Math.pow(L, 0.45));

  const width = 2048; 
  const height = 1024; 
  canvas.width = width;
  canvas.height = height;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = 'white';
  ctx.font = `bold ${Math.floor(fontSize)}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, height / 2);

  const imgData = ctx.getImageData(0, 0, width, height);
  const points: number[] = [];
  const step = 2; 
  
  const scale = 0.04; 

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const index = (y * width + x) * 4;
      if (imgData.data[index] > 128) {
        const px = (x - width / 2) * scale;
        const py = -(y - height / 2) * scale; // Invert Y for 3D
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
}

// Define the structure of a saved memory snapshot
interface MemorySnapshot {
  matrix: Float32Array;
  x: Float32Array;
  v: Float32Array; 
  phase: Float32Array; 
  spin: Float32Array; 
  target: Float32Array;
  hasTarget: Uint8Array;
  activation: Float32Array;
  regionID: Uint8Array;
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  
  // Track system state for adaptive controls (prevents React render loops)
  const systemState = useRef({
      meanError: 10.0, // Start with high error assumption
      formationProgress: 0.0,
      logTimer: 0
  });

  // We use dataRef passed from parent to share state with UI visualization
  const data = dataRef;

  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    // Initialize data if not already done or if size changes
    if (data.current.x.length !== count * 3) {
        data.current = {
            x: new Float32Array(count * 3),
            v: new Float32Array(count * 3),
            phase: new Float32Array(count),
            spin: new Float32Array(count),
            activation: new Float32Array(count),
            target: new Float32Array(count * 3),
            hasTarget: new Uint8Array(count),
            memoryMatrix: new Float32Array(memorySize).fill(-1), 
            regionID: new Uint8Array(count),
            forwardMatrix: new Float32Array(memorySize),
            feedbackMatrix: new Float32Array(memorySize),
            delayedActivation: new Float32Array(count),
            lastActiveTime: new Float32Array(count),
        };
        
        // Refined Fibonacci Sphere Distribution
        const phi = Math.PI * (3 - Math.sqrt(5)); // Golden Angle
        
        for (let i = 0; i < count; i++) {
            const y = 1 - (i + 0.5) * (2 / count); 
            const radiusAtY = Math.sqrt(1 - y * y); 
            const theta = phi * i; 
            
            const r = 14.0 * Math.cbrt(Math.random()); 
            
            data.current.x[i * 3] = Math.cos(theta) * radiusAtY * r;
            data.current.x[i * 3 + 1] = y * r;
            data.current.x[i * 3 + 2] = Math.sin(theta) * radiusAtY * r;

            data.current.phase[i] = Math.random() * Math.PI * 2;
            data.current.spin[i] = Math.random() > 0.5 ? 0.5 : -0.5;

            // Region Init (Sprint 0)
            if (i < Math.floor(count * 0.25)) data.current.regionID[i] = 0;      // Input A
            else if (i < Math.floor(count * 0.5)) data.current.regionID[i] = 1;  // Input B
            else data.current.regionID[i] = 2;                                   // Associative
        }
        
        memoryBank.current.clear();
        systemState.current.meanError = 10.0; 
    }

  }, [params.particleCount]);

  useEffect(() => {
    const { type, slot } = params.memoryAction;
    if (type === 'idle') return;

    if (type === 'save') {
        const currentMatrix = data.current.memoryMatrix;
        if (currentMatrix) {
            // Quantum Snapshot: Includes Phase and Spin for high-fidelity recall
            const snapshot: MemorySnapshot = {
                matrix: new Float32Array(data.current.memoryMatrix),
                x: new Float32Array(data.current.x),
                v: new Float32Array(data.current.v),
                phase: new Float32Array(data.current.phase),
                spin: new Float32Array(data.current.spin),
                target: new Float32Array(data.current.target),
                hasTarget: new Uint8Array(data.current.hasTarget),
                activation: new Float32Array(data.current.activation),
                regionID: new Uint8Array(data.current.regionID)
            };
            memoryBank.current.set(slot, snapshot);
            console.log(`[Memory] Snapshot Saved to Slot ${slot}. | Connections: ${snapshot.matrix.filter(v=>v!==-1).length}`);
        }
    } else if (type === 'load') {
        const snapshot = memoryBank.current.get(slot);
        if (snapshot && data.current.memoryMatrix) {
            // 1. Restore the Generative Model (Weights & Expectations)
            data.current.memoryMatrix.set(snapshot.matrix);
            data.current.target.set(snapshot.target);
            data.current.hasTarget.set(snapshot.hasTarget);
            
            // 2. Restore Quantum Internal State (Phase & Spin) - "Hidden States"
            data.current.phase.set(snapshot.phase);
            data.current.spin.set(snapshot.spin);
            data.current.activation.set(snapshot.activation);
            if(snapshot.regionID) data.current.regionID.set(snapshot.regionID);

            // 3. DO NOT Restore Position (x) directly.
            // By NOT setting x, we force the network to "Infer" the state via Free Energy Minimization.
            // This aligns with the PCN framework where dynamics drive the system to the attractor.
            // data.current.x.set(snapshot.x); // <--- REMOVED
            
            // 4. Inject Thermal Noise (Entropy) to facilitate transition
            // This simulates "fuzzy moving" / destabilization of the previous state.
            // Reduced thermal noise to help with wobble (0.5 -> 0.2)
            for(let i=0; i<data.current.v.length; i++) {
                data.current.v[i] = (Math.random() - 0.5) * 0.2; 
            }
            
            console.log(`[Memory] Loaded Slot ${slot}. Initiating Associative Recall.`);
        }
    }
  }, [params.memoryAction]);

  useEffect(() => {
    if (params.memoryResetTrigger > 0) {
      if (data.current.memoryMatrix) {
        data.current.memoryMatrix.fill(-1);
        console.log("[Memory] Working Memory Wiped (Entropy State).");
      }
    }
  }, [params.memoryResetTrigger]);

  useEffect(() => {
    const count = params.particleCount;
    const { positions, count: pointCount } = textToPoints(params.inputText);
    
    data.current.hasTarget.fill(0);

    if (pointCount > 0) {
        const currentPositions = data.current.x;
        
        // 1. Gather Target Points
        const targets: {x: number, y: number, z: number}[] = [];
        if (pointCount > count) {
            const step = pointCount / count;
            for (let i = 0; i < count; i++) {
                const idx = Math.floor(i * step);
                targets.push({
                    x: positions[idx * 3],
                    y: positions[idx * 3 + 1],
                    z: positions[idx * 3 + 2]
                });
            }
        } else {
            for (let i = 0; i < pointCount; i++) {
                targets.push({
                    x: positions[i * 3],
                    y: positions[i * 3 + 1],
                    z: positions[i * 3 + 2]
                });
            }
        }
        
        // 2. Spatial Indexing (Morton Codes / Z-Order Curve)
        const getMortonCode = (x: number, y: number, z: number) => {
            const map = (v: number) => Math.floor(Math.max(0, Math.min(1023, (v + 30) * 17)));
            
            let xx = map(x);
            let yy = map(y);
            let zz = map(z);

            xx = (xx | (xx << 16)) & 0x030000FF;
            xx = (xx | (xx <<  8)) & 0x0300F00F;
            xx = (xx | (xx <<  4)) & 0x030C30C3;
            xx = (xx | (xx <<  2)) & 0x09249249;

            yy = (yy | (yy << 16)) & 0x030000FF;
            yy = (yy | (yy <<  8)) & 0x0300F00F;
            yy = (yy | (yy <<  4)) & 0x030C30C3;
            yy = (yy | (yy <<  2)) & 0x09249249;

            zz = (zz | (zz << 16)) & 0x030000FF;
            zz = (zz | (zz <<  8)) & 0x0300F00F;
            zz = (zz | (zz <<  4)) & 0x030C30C3;
            zz = (zz | (zz <<  2)) & 0x09249249;

            return xx | (yy << 1) | (zz << 2);
        };

        // Filter particles based on active region
        const availableIndices = [];
        for (let i = 0; i < count; i++) {
            if (params.targetRegion === -1 || data.current.regionID[i] === params.targetRegion) {
                availableIndices.push(i);
            }
        }
        
        const activeCount = Math.min(availableIndices.length, targets.length);

        const pIndices = availableIndices.map(i => ({
            index: i,
            code: getMortonCode(currentPositions[i*3], currentPositions[i*3+1], currentPositions[i*3+2])
        }));

        const tIndices = targets.map((t, i) => ({
            index: i,
            code: getMortonCode(t.x, t.y, t.z)
        }));

        pIndices.sort((a, b) => a.code - b.code);
        tIndices.sort((a, b) => a.code - b.code);

        // 3. Initial Assignment based on Spatial Sort
        for (let i = 0; i < activeCount; i++) {
            const pid = pIndices[i].index;
            const t = targets[tIndices[i].index];
            
            data.current.target[pid * 3] = t.x;
            data.current.target[pid * 3 + 1] = t.y;
            data.current.target[pid * 3 + 2] = t.z;
            data.current.hasTarget[pid] = 1;
            data.current.activation[pid] = 1.0;
        }

        // 4. Enhanced Iterative Refinement
        const iterations = activeCount * 128; 
        
        for (let k = 0; k < iterations; k++) {
            const i1 = Math.floor(Math.random() * activeCount);
            const i2 = Math.floor(Math.random() * activeCount);
            
            if (i1 === i2) continue;

            const pidA = pIndices[i1].index;
            const pidB = pIndices[i2].index;
            
            const pax = currentPositions[pidA * 3], pay = currentPositions[pidA * 3 + 1], paz = currentPositions[pidA * 3 + 2];
            const pbx = currentPositions[pidB * 3], pby = currentPositions[pidB * 3 + 1], pbz = currentPositions[pidB * 3 + 2];

            const tax = data.current.target[pidA * 3], tay = data.current.target[pidA * 3 + 1], taz = data.current.target[pidA * 3 + 2];
            const tbx = data.current.target[pidB * 3], tby = data.current.target[pidB * 3 + 1], tbz = data.current.target[pidB * 3 + 2];

            const costCurrent = (pax-tax)**2 + (pay-tay)**2 + (paz-taz)**2 + 
                                (pbx-tbx)**2 + (pby-tby)**2 + (pbz-tbz)**2;
            
            const costSwap = (pax-tbx)**2 + (pay-tby)**2 + (paz-tbz)**2 + 
                             (pbx-tax)**2 + (pby-tay)**2 + (pbz-taz)**2;

            if (costSwap < costCurrent) {
                data.current.target[pidA * 3] = tbx;
                data.current.target[pidA * 3 + 1] = tby;
                data.current.target[pidA * 3 + 2] = tbz;
                
                data.current.target[pidB * 3] = tax;
                data.current.target[pidB * 3 + 1] = tay;
                data.current.target[pidB * 3 + 2] = taz;
            }
        }
    }
    
    systemState.current.meanError = 10.0;

    if (ghostRef.current) {
      const dummyObj = new THREE.Object3D();
      dummyObj.scale.set(0,0,0);
      for(let k=0; k<count; k++) ghostRef.current.setMatrixAt(k, dummyObj.matrix);

      for (let i = 0; i < count; i++) {
         if (data.current.hasTarget[i]) {
            const tx = data.current.target[i * 3];
            const ty = data.current.target[i * 3 + 1];
            const tz = data.current.target[i * 3 + 2];
            
            dummyObj.position.set(tx, ty, tz);
            dummyObj.scale.set(0.15, 0.15, 0.15);
            dummyObj.rotation.set(0,0,0);
            dummyObj.updateMatrix();
            ghostRef.current.setMatrixAt(i, dummyObj.matrix);
         }
      }
      ghostRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [params.inputText, params.particleCount, params.targetRegion]);

  const maxConnections = params.particleCount * 8;
  const lineGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(maxConnections * 2 * 3);
    const col = new Float32Array(maxConnections * 2 * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return geo;
  }, [maxConnections]);

  useFrame((state) => {
    if (!meshRef.current || !linesRef.current) return;
    
    // PLAY/PAUSE LOGIC
    if (params.paused) {
        // Even when paused, ensure visuals match current state
        meshRef.current.instanceMatrix.needsUpdate = true;
        return;
    }

    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, spatialLearningRate, dataGravity, plasticity, damping } = params;
    const count = params.particleCount;
    // Updated Destructuring to include regionID (Sprint 0)
    const { x, v, phase, activation, target, hasTarget, memoryMatrix, regionID } = data.current;

    // Safety Check
    if (x.length === 0 || regionID.length === 0) return;
    
    let activeTargetCount = 0;
    for(let k = 0; k < count; k++) {
        if(hasTarget[k] === 1) activeTargetCount++;
    }
    const hasActiveTargets = activeTargetCount > 0;

    // --- 1. Global Statistics Calculation ---
    let totalActivation = 0;
    let totalRadiusSq = 0;
    let totalVelocitySq = 0; 

    for (let k = 0; k < count; k++) {
      totalActivation += activation[k];
      const px = x[k * 3];
      const py = x[k * 3 + 1];
      const pz = x[k * 3 + 2];
      totalRadiusSq += px * px + py * py + pz * pz;

      const vx = v[k * 3];
      const vy = v[k * 3 + 1];
      const vz = v[k * 3 + 2];
      totalVelocitySq += vx * vx + vy * vy + vz * vz;
    }
    
    const meanActivation = count > 0 ? totalActivation / count : 0;
    const meanRadius = count > 0 ? Math.sqrt(totalRadiusSq / count) : 8.0;
    const meanVelocitySq = count > 0 ? totalVelocitySq / count : 0;
    
    // Logging Throttler
    systemState.current.logTimer += 1;
    if (systemState.current.logTimer > 120 && params.memoryAction.type === 'idle') { // Every ~2 seconds
        // Only log during recall to debug wobble
        if (!hasActiveTargets && meanVelocitySq > 0.01) {
             console.debug(`[Physics] Recall Stability - Mean Kinetic Energy: ${meanVelocitySq.toFixed(5)}`);
        }
        systemState.current.logTimer = 0;
    }

    // --- 2. Adaptive Parameters (Physics Engine) ---
    
    const currentError = systemState.current.meanError;
    const formationProgress = Math.max(0, Math.min(1.0, 1.0 - (currentError / 12.0)));
    systemState.current.formationProgress = formationProgress;

    // Adaptive Learning Rate
    let adaptiveRateScale = 1.0;

    if (hasActiveTargets) {
        if (currentError > 4.0) {
            adaptiveRateScale = 1.2 + Math.min(1.8, (currentError - 4.0) * 0.15);
        } else if (currentError > 0.8) {
            adaptiveRateScale = 0.3 + (currentError - 0.8) * 0.28; 
        } else {
            adaptiveRateScale = 0.05 + currentError * 0.3;
        }
    } else {
         adaptiveRateScale = 1.0 / (1.0 + meanVelocitySq * 0.2);
    }

    const effectiveLearningRate = spatialLearningRate * adaptiveRateScale;

    // --- Density-Adaptive Equilibrium ---
    const nominalRadius = 8.0;
    const nominalCount = 800.0;
    const currentVol = Math.max(1.0, Math.pow(meanRadius, 3));
    const nominalVol = Math.pow(nominalRadius, 3);
    
    const relativeDensity = (count / currentVol) / (nominalCount / nominalVol);
    const densityScale = 1.0 / Math.pow(relativeDensity, 0.333);
    const clampedDensityScale = Math.max(0.5, Math.min(2.0, densityScale));
    
    const effectiveEquilibrium = equilibriumDistance * clampedDensityScale;

    // Adaptive Stiffness & Gravity
    const isLearning = plasticity > 0;
    
    let adaptiveStiffness = stiffness;
    let adaptiveGravity = dataGravity;

    if (hasActiveTargets) {
        adaptiveStiffness = stiffness * 0.005; // Lower stiffness so they don't fight the target text
        const baseGravity = Math.max(dataGravity, 0.1);
        adaptiveGravity = baseGravity * (isLearning ? 6.0 : 3.0);
        // TIGHTENING TEXT: Increase gravity force specifically when targets are present to make text sharper
        adaptiveGravity *= 2.0; 
    } else {
        adaptiveStiffness = isLearning ? stiffness * 0.2 : stiffness;
        adaptiveGravity = isLearning ? Math.max(dataGravity, 0.6) : dataGravity;
    }

    // Adaptive Damping (WOBBLE FIX)
    let effectiveDamping = params.damping; // Default ~0.8

    if (hasActiveTargets) {
        // Input Mode: High friction to freeze them on the letters
        effectiveDamping = isLearning ? 0.5 : 0.8; 
    } else {
        // RECALL MODE:
        // PREVIOUSLY: minRetention = 0.92 (Slippery/Ice) -> Caused Wobble
        // FIX: Lower retention = Higher friction/damping = Honey
        const wobbleSuppression = 0.65; 
        effectiveDamping = wobbleSuppression;
    }

    // Dynamic Visualization Threshold
    const systemStress = Math.min(1.0, meanActivation);
    const systemKinetic = Math.min(1.0, meanVelocitySq * 0.5); 
    const excitation = Math.min(1.0, systemStress * 0.5 + systemKinetic * 0.5);
    
    const minThreshold = 0.15;
    const maxThreshold = 0.60;
    const connectionThreshold = minThreshold + (maxThreshold - minThreshold) * (excitation * excitation);

    let lineIndex = 0;
    const linePositions = linesRef.current.geometry.attributes.position.array as Float32Array;
    const lineColors = linesRef.current.geometry.attributes.color.array as Float32Array;
    
    let frameTotalDist = 0;
    let frameActiveTargets = 0;

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0;
      let stress = 0;

      const ix = x[i * 3];
      const iy = x[i * 3 + 1];
      const iz = x[i * 3 + 2];
      
      const ri = regionID[i]; // Sprint 0: Get Region

      for (let j = 0; j < count; j++) {
        if (i === j) continue;

        // Sprint 0: Inhibition & Gating Logic
        const rj = regionID[j];
        
        // 1. Inhibition: Block direct Input-Input coupling (0 <-> 1)
        if ((ri === 0 && rj === 1) || (ri === 1 && rj === 0)) continue;

        const jx = x[j * 3];
        const jy = x[j * 3 + 1];
        const jz = x[j * 3 + 2];

        const dx = jx - ix;
        const dy = jy - iy;
        const dz = jz - iz;
        const distSq = dx * dx + dy * dy + dz * dz;

        const dist = Math.sqrt(distSq);
        if (dist < 0.001 || dist > couplingDecay * 1.5) continue; 

        const phaseDiff = phase[j] - phase[i];
        
        const spatialWeight = Math.exp(-distSq / (couplingDecay * couplingDecay));
        const phaseWeight = (1.0 + Math.cos(phaseDiff)) * 0.5;
        
        const couplingProb = spatialWeight * phaseWeight;

        const memIndex = i * count + j;
        let r0 = memoryMatrix[memIndex];
        const isLearned = r0 !== -1;

        if (isLearning && couplingProb > 0.05) {
            if (r0 === -1) r0 = dist; 
            // Clamp r0 to stop it from drifting during vibration
            // Only update if we are very stable or first learn
            if (meanVelocitySq < 0.05 || r0 === dist) {
                 r0 = r0 + (dist - r0) * plasticity;
            }
            memoryMatrix[memIndex] = r0;

            phaseDelta += Math.sin(phaseDiff) * plasticity * 2.0;
        }

        if (r0 === -1) r0 = effectiveEquilibrium;

        // 3. Force Calculation
        let localStiffness = adaptiveStiffness;
        if (dist < r0) localStiffness = Math.max(stiffness, 2.0); 

        // Sprint 0: Sensory Rigidity - Harden intra-region bonds for inputs
        if (ri === rj && (ri === 0 || ri === 1)) {
            localStiffness *= 5.0; 
        }
        
        // Recall Stability: If learned connection, boost stiffness to hold shape
        if (!hasActiveTargets && isLearned) {
            localStiffness *= 1.5;
        }

        const effectiveForce = localStiffness * (dist - r0) * couplingProb;
        
        stress += Math.abs(effectiveForce);

        const nx = dx / dist;
        const ny = dy / dist;
        const nz = dz / dist;

        fx += nx * effectiveForce;
        fy += ny * effectiveForce;
        fz += nz * effectiveForce;

        if (couplingProb > 0.1) phaseDelta += couplingProb * Math.sin(phaseDiff);

        // Visualization
        const showLine = isLearned || (couplingProb > connectionThreshold);
        
        if (j > i && showLine && lineIndex < maxConnections) {
          
          linePositions[lineIndex * 6] = ix;
          linePositions[lineIndex * 6 + 1] = iy;
          linePositions[lineIndex * 6 + 2] = iz;

          linePositions[lineIndex * 6 + 3] = jx;
          linePositions[lineIndex * 6 + 4] = jy;
          linePositions[lineIndex * 6 + 5] = jz;

          let r=0.1, g=0.5, b=0.7; 
          let alpha = 0.0;

          if (isLearned) {
             r=1.0; g=0.84; b=0.0; 
             const err = Math.abs(dist - r0);
             alpha = Math.max(0.3, 1.0 - err); 
          } else {
             const baseR = 0.1; const baseG = 0.2; const baseB = 0.6;
             const hotR = 0.8; const hotG = 0.9; const hotB = 1.0;
             
             r = baseR + (hotR - baseR) * excitation;
             g = baseG + (hotG - baseG) * excitation;
             b = baseB + (hotB - baseB) * excitation;
             
             alpha = Math.sqrt(Math.max(0, couplingProb - connectionThreshold) / (1.0 - connectionThreshold));
          }

          lineColors[lineIndex * 6] = r * alpha;
          lineColors[lineIndex * 6 + 1] = g * alpha;
          lineColors[lineIndex * 6 + 2] = b * alpha;
          
          lineColors[lineIndex * 6 + 3] = r * alpha;
          lineColors[lineIndex * 6 + 4] = g * alpha;
          lineColors[lineIndex * 6 + 5] = b * alpha;

          lineIndex++;
        }
      }

      // ** GRAVITATION & CONTROL LOGIC **
      if (hasTarget[i]) {
        const tx = target[i * 3];
        const ty = target[i * 3 + 1];
        const tz = target[i * 3 + 2];

        const dx = tx - ix;
        const dy = ty - iy;
        const dz = tz - iz;
        const distToTarget = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        frameTotalDist += distToTarget;
        frameActiveTargets++;
        
        const dirX = dx / (distToTarget + 0.0001);
        const dirY = dy / (distToTarget + 0.0001);
        const dirZ = dz / (distToTarget + 0.0001);

        let forceMag = adaptiveGravity; 
        
        if (distToTarget > 0.5) {
            const distFactor = 1.0 + Math.pow(distToTarget, 0.6); 
            forceMag *= distFactor;
        } else {
             // TIGHTENING TEXT: Boost close-range gravity significantly
             forceMag *= (distToTarget * 8.0); // Increased from 6.0
        }
        
        fx += dirX * forceMag;
        fy += dirY * forceMag;
        fz += dirZ * forceMag;
        
        const vx = v[i*3], vy = v[i*3+1], vz = v[i*3+2];
        const speedSq = vx*vx + vy*vy + vz*vz;
        const maxSpeed = 1.0; 
        if (speedSq > maxSpeed * maxSpeed) {
            const scale = maxSpeed / Math.sqrt(speedSq);
            v[i*3] *= scale;
            v[i*3+1] *= scale;
            v[i*3+2] *= scale;
        }
      }

      activation[i] = activation[i] * 0.95 + stress * 0.05;

      v[i * 3] = v[i * 3] * effectiveDamping + fx * effectiveLearningRate;
      v[i * 3 + 1] = v[i * 3 + 1] * effectiveDamping + fy * effectiveLearningRate;
      v[i * 3 + 2] = v[i * 3 + 2] * effectiveDamping + fz * effectiveLearningRate;

      x[i * 3] += v[i * 3];
      x[i * 3 + 1] += v[i * 3 + 1];
      x[i * 3 + 2] += v[i * 3 + 2];

      const rSq = x[i * 3]**2 + x[i * 3 + 1]**2 + x[i * 3 + 2]**2;
      if (rSq > 900) {
           const scale = 0.98;
           x[i * 3] *= scale;
           x[i * 3 + 1] *= scale;
           x[i * 3 + 2] *= scale;
      }

      phase[i] += phaseSyncRate * phaseDelta + 0.02;

      TEMP_OBJ.position.set(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
      
      const energyLevel = Math.min(1.0, activation[i]); 
      const size = 0.4 + 0.4 * energyLevel;
      
      TEMP_OBJ.scale.set(size, size, size);
      TEMP_OBJ.updateMatrix();
      meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      // Sprint 0: Visual Coloring by Region
      const region = regionID[i];
      let baseR = 0, baseG = 0, baseB = 0;
      
      if (region === 0) { // Cyan (Input A)
          baseR = 0.0; baseG = 1.0; baseB = 1.0;
      } else if (region === 1) { // Pink (Input B)
          baseR = 1.0; baseG = 0.4; baseB = 0.8; 
      } else { // Gold (Associative)
          baseR = 1.0; baseG = 0.84; baseB = 0.0;
      }
      
      const phaseMod = Math.sin(phase[i]) * 0.1;
      // Blend base color with activation intensity
      const r = Math.min(1, baseR + energyLevel * 0.4 + phaseMod);
      const g = Math.min(1, baseG + energyLevel * 0.4 + phaseMod);
      const b = Math.min(1, baseB + energyLevel * 0.4 + phaseMod);

      TEMP_COLOR.setRGB(r, g, b);
      meshRef.current.setColorAt(i, TEMP_COLOR);
    }
    
    if (frameActiveTargets > 0) {
        systemState.current.meanError = frameTotalDist / frameActiveTargets;
    } else {
        systemState.current.meanError = 0;
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    
    linesRef.current.geometry.setDrawRange(0, lineIndex * 2);
    linesRef.current.geometry.attributes.position.needsUpdate = true;
    linesRef.current.geometry.attributes.color.needsUpdate = true;
  });

  return (
    <>
      <InstancedMesh ref={ghostRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[1, 8, 8]} />
        <MeshBasicMaterial color="#ffffff" transparent opacity={0.1} wireframe />
      </InstancedMesh>

      <InstancedMesh ref={meshRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[0.3, 16, 16]} />
        <MeshStandardMaterial roughness={0.2} metalness={0.8} />
      </InstancedMesh>

      <LineSegments ref={linesRef} geometry={lineGeometry}>
        <LineBasicMaterial vertexColors={true} transparent opacity={0.4} blending={THREE.AdditiveBlending} depthWrite={false} />
      </LineSegments>
    </>
  );
};

// --- Memory Matrix Heatmap ---

interface MemoryHeatmapProps {
    dataRef: React.MutableRefObject<ParticleData>;
    params: SimulationParams;
}

const MemoryHeatmap: React.FC<MemoryHeatmapProps> = ({ dataRef, params }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let animationFrameId: number;

        const render = () => {
            if (!canvasRef.current || !dataRef.current.memoryMatrix) return;
            
            const count = params.particleCount;
            // Optimization: Only update visually every X frames if needed, but modern canvas is fast.
            // We'll update every frame for smoothness.
            
            const ctx = canvasRef.current.getContext('2d');
            if (!ctx) return;
            
            const size = 256; // Fixed visual size for the canvas buffer (scaled up by CSS)
            if (canvasRef.current.width !== size) {
                canvasRef.current.width = size;
                canvasRef.current.height = size;
            }

            const matrix = dataRef.current.memoryMatrix;
            const imgData = ctx.createImageData(size, size);
            const pixels = new Uint32Array(imgData.data.buffer);
            
            // To fit N particles into 'size' pixels, we might need to downsample or just display a subset.
            // Visualizing 800x800 is possible but requires a larger canvas buffer.
            // Let's map particle index (0..count) to pixel coordinates (x, y) via simple wrapping
            // or just sample the matrix sparsely.
            // Better approach for visualization: Map the Interaction Matrix.
            // Since N is 800, we can sample the first 256x256 interactions, or scale down.
            
            // Let's do a direct mapping of a subset: Top-Left 256x256 particles.
            // This shows the local connectivity of the first chunk of the cloud.
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const idx = (y * size + x);
                    
                    // Map x,y visual coordinates to particle indices.
                    // We step through the particles.
                    const pI = Math.floor((y / size) * count);
                    const pJ = Math.floor((x / size) * count);
                    
                    // Actually, simpler: just show index i vs index j for i,j < size
                    // If count > size, we only show interactions between first 256 particles.
                    // If we want to show ALL, we need to bin.
                    
                    // Let's just show the first 256x256 connections to keep it performant and crisp.
                    // Most particles behave similarly.
                    
                    const matIdx = y * count + x; // Linear index in the full matrix
                    
                    if (y < count && x < count) {
                         const val = matrix[matIdx];
                         
                         if (val !== -1) {
                             // Connection exists.
                             // Color logic: ABGR (Little Endian)
                             // Gold: Alpha=FF, Blue=00, Green=D7, Red=FF -> 0xFF00D7FF
                             pixels[idx] = 0xFF00D7FF;
                         } else {
                             // Black background: Alpha=FF, B=0, G=0, R=0 -> 0xFF000000
                             // Or transparent: 0x00000000
                             pixels[idx] = 0xFF101010; // Very dark gray
                         }
                    } else {
                        pixels[idx] = 0xFF000000;
                    }
                }
            }
            
            ctx.putImageData(imgData, 0, 0);
            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [params.particleCount]);

    return (
        <div ref={containerRef} className="mt-4 border border-cyan-500/30 rounded p-2 bg-black/60">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-cyan-300 mb-2">Memory Matrix (Subsection)</h3>
            <canvas 
                ref={canvasRef} 
                className="w-full h-auto aspect-square image-pixelated rounded border border-white/5"
            />
            <p className="text-[9px] text-gray-500 mt-1">Real-time synaptic weights (Yellow = Learned)</p>
        </div>
    );
};

// --- UI Overlay ---

interface UIOverlayProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  dataRef: React.MutableRefObject<ParticleData>;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams, dataRef }) => {
  const [localText, setLocalText] = useState("");
  const [showInfo, setShowInfo] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [activeTab, setActiveTab] = useState<'auto' | 'manual'>('auto');
  
  // Auto-Run State
  const [autoRunPhase, setAutoRunPhase] = useState<'idle' | 'reset' | 'inject' | 'stabilize' | 'learning' | 'saving' | 'forgetting' | 'recalling'>('idle');
  const [autoStatus, setAutoStatus] = useState("");

  const handleChange = (key: keyof SimulationParams, value: number | string | object | boolean) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const togglePause = () => {
    handleChange('paused', !params.paused);
  };

  const reset = () => {
    setParams(DEFAULT_PARAMS);
    setLocalText("");
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleChange('inputText', localText);
  };
  
  const clearInput = () => {
    setLocalText("");
    handleChange('inputText', "");
  };

  const forgetCurrentMemory = () => {
    handleChange('memoryResetTrigger', params.memoryResetTrigger + 1);
  };

  const togglePlasticity = (active: boolean) => {
      if (active) {
          setParams(prev => ({
              ...prev,
              plasticity: 0.1,
              damping: 0.95, 
              dataGravity: Math.max(prev.dataGravity, 0.5) 
          }));
      } else {
          setParams(prev => ({
              ...prev,
              plasticity: 0,
              damping: 0.85, 
          }));
      }
  };

  const handleMemoryAction = (type: 'save' | 'load', slot: number) => {
    setParams(prev => ({
      ...prev,
      memoryAction: { type, slot, triggerId: prev.memoryAction.triggerId + 1 }
    }));
  };

  // --- Auto-Run Orchestration ---
  useEffect(() => {
    if (autoRunPhase === 'idle') return;

    let timer: ReturnType<typeof setTimeout>;

    if (autoRunPhase === 'reset') {
        setAutoStatus("Phase 1: System Reset & Calibration");
        setParams(DEFAULT_PARAMS);
        setLocalText("");
        timer = setTimeout(() => setAutoRunPhase('inject'), 1500);
    } 
    else if (autoRunPhase === 'inject') {
        const word = "Q-MIND";
        setAutoStatus(`Phase 2: Injecting Data Signal "${word}"`);
        setLocalText(word);
        setParams(prev => ({ ...prev, inputText: word, dataGravity: 0.4 }));
        console.log("[AutoPilot] Injecting Data...");
        timer = setTimeout(() => setAutoRunPhase('stabilize'), 3000); // Increased
    }
    else if (autoRunPhase === 'stabilize') {
        setAutoStatus("Phase 3: Stabilizing Particle Cloud...");
        // Just waiting for physics to settle visually
        console.log("[AutoPilot] Stabilizing...");
        timer = setTimeout(() => setAutoRunPhase('learning'), 3500); // Increased
    }
    else if (autoRunPhase === 'learning') {
        setAutoStatus("Phase 4: Enabling Plasticity (Hebbian Learning)");
        togglePlasticity(true);
        console.log("[AutoPilot] Learning Active.");
        // Allow time for yellow lines to form
        timer = setTimeout(() => setAutoRunPhase('saving'), 5000); // Increased
    }
    else if (autoRunPhase === 'saving') {
        setAutoStatus("Phase 5: Persisting Quantum State to Memory Slot 1");
        handleMemoryAction('save', 1);
        timer = setTimeout(() => setAutoRunPhase('forgetting'), 1500);
    }
    else if (autoRunPhase === 'forgetting') {
        setAutoStatus("Phase 6: Inducing Entropy (Forgetting Input)");
        togglePlasticity(false);
        setLocalText("");
        setParams(prev => ({ ...prev, inputText: "", memoryResetTrigger: prev.memoryResetTrigger + 1 }));
        timer = setTimeout(() => setAutoRunPhase('recalling'), 3000);
    }
    else if (autoRunPhase === 'recalling') {
        setAutoStatus("Phase 7: Holographic Associative Recall");
        handleMemoryAction('load', 1);
        timer = setTimeout(() => {
            setAutoStatus("Experiment Complete.");
            setAutoRunPhase('idle');
        }, 5000);
    }

    return () => clearTimeout(timer);
  }, [autoRunPhase]);

  const startAutoRun = () => {
      setAutoRunPhase('reset');
  };

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end">
      <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded-xl p-5 text-white w-full shadow-2xl pointer-events-auto overflow-y-auto max-h-[90vh] custom-scrollbar">
        
        <div className="flex justify-between items-start mb-4">
            <div>
                <h1 className="text-xl font-bold text-cyan-400">Holographic Memory</h1>
                <p className="text-[10px] text-gray-400 italic">Associative Storage in Particle Clouds</p>
            </div>
            <button 
                onClick={() => setShowHelp(true)}
                className="text-cyan-400 hover:text-cyan-200 transition-colors"
                title="Open Simulation Guide"
            >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
                </svg>
            </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex bg-gray-900 rounded p-1 mb-4 border border-gray-700">
            <button 
                onClick={() => setActiveTab('auto')}
                className={`flex-1 py-1 text-xs font-bold rounded transition-colors ${activeTab === 'auto' ? 'bg-cyan-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
                Auto-Pilot
            </button>
            <button 
                onClick={() => setActiveTab('manual')}
                className={`flex-1 py-1 text-xs font-bold rounded transition-colors ${activeTab === 'manual' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
                Manual Control
            </button>
        </div>
        
        {/* Play / Pause Toggle */}
        <div className="bg-white/5 p-2 rounded mb-4 flex justify-center border border-white/10">
             <button
                onClick={togglePause}
                className={`flex items-center gap-2 px-6 py-2 rounded-full border transition-all ${params.paused ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50' : 'bg-green-500/20 text-green-300 border-green-500/50'} `}
             >
                {params.paused ? (
                   <>
                     <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                       <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                     </svg>
                     <span className="text-xs font-bold uppercase">Resume Physics</span>
                   </>
                ) : (
                   <>
                     <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
                     </svg>
                     <span className="text-xs font-bold uppercase">Pause Simulation</span>
                   </>
                )}
             </button>
        </div>

        {/* AUTO-PILOT VIEW */}
        {activeTab === 'auto' && (
            <div className="animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="bg-cyan-900/10 border border-cyan-500/30 rounded-lg p-4 mb-4 text-center">
                    <div className="w-16 h-16 bg-cyan-500/10 rounded-full flex items-center justify-center mx-auto mb-3 border border-cyan-500/30">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 text-cyan-400">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                        </svg>
                    </div>
                    <h3 className="text-cyan-300 font-bold mb-1">Run Simulation</h3>
                    <p className="text-[10px] text-gray-400 mb-4">Automated training & recall cycle</p>
                    
                    {autoRunPhase === 'idle' ? (
                        <button 
                            onClick={startAutoRun}
                            className="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded shadow-lg shadow-cyan-500/20 transition-all active:scale-95"
                        >
                            Start Experiment
                        </button>
                    ) : (
                        <div className="text-left bg-black/40 p-3 rounded border border-cyan-500/20">
                             <div className="flex items-center gap-2 mb-2">
                                <span className="relative flex h-2 w-2">
                                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                                </span>
                                <span className="text-[10px] text-cyan-300 uppercase tracking-wider font-bold">Running</span>
                             </div>
                             <p className="text-xs text-white font-mono leading-tight">{autoStatus}</p>
                        </div>
                    )}
                </div>
                
                <div className="text-[10px] text-gray-500 space-y-1 pl-2 border-l border-gray-700">
                    <p>Sequence:</p>
                    <p>1. Inject Data Pattern</p>
                    <p>2. Enable Hebbian Learning</p>
                    <p>3. Persist Synaptic Matrix</p>
                    <p>4. Induce Chaos (Forget)</p>
                    <p>5. Associative Recall</p>
                </div>
            </div>
        )}
        
        {/* MANUAL VIEW */}
        {activeTab === 'manual' && (
            <div className="animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="mb-4 bg-cyan-900/20 p-3 rounded-lg border border-cyan-500/30">
                  <h2 className="text-[10px] font-bold uppercase tracking-widest text-cyan-300 mb-2">1. Encode Data</h2>
                  <form onSubmit={handleTextSubmit} className="flex gap-2 mb-2">
                    <input 
                      type="text" 
                      value={localText}
                      onChange={(e) => setLocalText(e.target.value)}
                      placeholder="e.g. QUBIT"
                      className="w-full bg-black/50 border border-cyan-500/50 rounded px-2 py-1 text-sm focus:outline-none focus:border-cyan-400 font-mono"
                      maxLength={16} 
                    />
                    <button type="submit" className="bg-cyan-600 hover:bg-cyan-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors">
                      INPUT
                    </button>
                  </form>

                  {/* New Region Selector for Sprint 0 Verification */}
                  <div className="flex gap-1 mb-2">
                    <button 
                        onClick={() => handleChange('targetRegion', 0)}
                        className={`flex-1 text-[9px] py-1 rounded border border-cyan-500/30 ${params.targetRegion === 0 ? 'bg-cyan-500 text-black font-bold' : 'bg-black/30 text-cyan-500'}`}
                    >
                        Input A
                    </button>
                    <button 
                        onClick={() => handleChange('targetRegion', 1)}
                        className={`flex-1 text-[9px] py-1 rounded border border-pink-500/30 ${params.targetRegion === 1 ? 'bg-pink-500 text-black font-bold' : 'bg-black/30 text-pink-500'}`}
                    >
                        Input B
                    </button>
                     <button 
                        onClick={() => handleChange('targetRegion', -1)}
                        className={`flex-1 text-[9px] py-1 rounded border border-gray-500/30 ${params.targetRegion === -1 ? 'bg-gray-200 text-black font-bold' : 'bg-black/30 text-gray-400'}`}
                    >
                        All
                    </button>
                  </div>
                  
                  <div className="mb-2 pt-2 border-t border-cyan-500/20">
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="font-mono text-cyan-200">Data Gravity</span>
                      <span className="text-cyan-300">{params.dataGravity.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.01"
                      value={params.dataGravity}
                      onChange={(e) => handleChange('dataGravity', parseFloat(e.target.value))}
                      className="w-full h-1 bg-cyan-900/50 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                    />
                  </div>

                  {params.inputText && (
                       <button onClick={clearInput} className="w-full bg-red-500/20 hover:bg-red-500/40 text-red-200 text-[10px] py-1 rounded transition-colors">
                         Remove Input (Enter Recall Mode)
                       </button>
                  )}
                </div>

                <div className="mb-4 bg-yellow-900/20 p-3 rounded-lg border border-yellow-500/30">
                    <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                            <span className="text-yellow-300 text-[10px] font-bold uppercase tracking-wider">2. Imprint (Learn)</span>
                            <button 
                                onClick={() => setShowInfo(!showInfo)}
                                className="text-yellow-500 hover:text-yellow-300 transition-colors"
                            >
                                <span className="text-xs border border-yellow-500 rounded-full w-4 h-4 flex items-center justify-center">?</span>
                            </button>
                        </div>
                        <div className={`w-2 h-2 rounded-full ${params.plasticity > 0 ? 'bg-yellow-400 animate-pulse' : 'bg-gray-600'}`}></div>
                    </div>
                    
                    {showInfo && (
                        <div className="mb-3 p-2 bg-black/40 rounded border border-yellow-500/20 text-[10px] text-gray-300 leading-relaxed shadow-inner">
                            <p className="mb-1"><strong>Plasticity:</strong> Hardens connections between particles based on their current positions.</p>
                        </div>
                    )}

                    <div className="flex items-center gap-2 mb-2">
                       <button 
                          onClick={() => togglePlasticity(true)} 
                          className={`flex-1 py-1 rounded text-xs font-bold transition-colors ${params.plasticity > 0 ? 'bg-yellow-500 text-black' : 'bg-gray-700 text-gray-300'}`}
                       >
                          ON
                       </button>
                       <button 
                          onClick={() => togglePlasticity(false)} 
                          className={`flex-1 py-1 rounded text-xs font-bold transition-colors ${params.plasticity === 0 ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-500'}`}
                       >
                          OFF
                       </button>
                    </div>
                    <p className="text-[9px] text-yellow-500/60 text-center italic">
                        {params.plasticity > 0 ? "Optimizing parameters for learning..." : "Standard physics active"}
                    </p>
                </div>

                <div className="mb-4 bg-purple-900/20 p-3 rounded-lg border border-purple-500/30">
                     <h2 className="text-[10px] font-bold uppercase tracking-widest text-purple-300 mb-3">3. Associative Memory Bank</h2>
                     
                     <div className="space-y-2">
                        {[1, 2, 3, 4].map(slot => (
                            <div key={slot} className="flex items-center justify-between gap-2">
                                <span className="text-xs text-purple-200 font-mono">Slot {slot}</span>
                                <div className="flex gap-1">
                                    <button 
                                        onClick={() => handleMemoryAction('save', slot)}
                                        className="bg-purple-700/50 hover:bg-purple-600 text-[10px] px-2 py-1 rounded text-purple-100 border border-purple-500/50"
                                    >
                                        Save
                                    </button>
                                     <button 
                                        onClick={() => handleMemoryAction('load', slot)}
                                        className="bg-purple-500 hover:bg-purple-400 text-[10px] px-3 py-1 rounded text-white font-bold"
                                    >
                                        Recall
                                    </button>
                                </div>
                            </div>
                        ))}
                     </div>
                     
                     <button 
                       onClick={forgetCurrentMemory} 
                       className="w-full mt-3 py-1 bg-white/5 hover:bg-white/10 text-gray-400 border border-white/10 rounded transition-colors text-[10px] uppercase tracking-wider"
                    >
                      Clear Current State
                    </button>
                </div>
            </div>
        )}

        {/* Legend for Regions */}
        <div className="mb-4 p-2 bg-white/5 rounded border border-white/10 flex justify-between text-[9px] text-gray-400">
             <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-400"></span> Input A</div>
             <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-pink-400"></span> Input B</div>
             <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400"></span> Assoc</div>
        </div>

        {/* Heatmap Visualization Added Here */}
        <MemoryHeatmap dataRef={dataRef} params={params} />

        <div className="h-px bg-gray-700 mb-4" />

        <div className="space-y-4">
          <button 
            onClick={reset}
            className="w-full py-2 bg-red-900/20 hover:bg-red-900/40 text-red-300 border border-red-500/20 rounded transition-colors text-[10px] uppercase tracking-wider"
          >
            Hard Reset
          </button>
        </div>
      </div>
      
      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 pointer-events-auto">
            <div className="bg-gray-900 border border-cyan-500/50 rounded-xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl relative">
                <button 
                    onClick={() => setShowHelp(false)}
                    className="absolute top-4 right-4 text-gray-400 hover:text-white"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
                
                <div className="p-6 overflow-y-auto custom-scrollbar">
                    <h2 className="text-2xl font-bold text-cyan-400 mb-4">Simulation Guide</h2>
                    
                    <div className="space-y-6 text-gray-300 text-sm">
                        <section>
                            <h3 className="text-lg font-bold text-white mb-2">The Concept</h3>
                            <p className="leading-relaxed">This simulation demonstrates <strong>Predictive Coding</strong> in a 3D particle system. Particles minimize "Free Energy" (difference between their state and a target state) to self-organize. By introducing plasticity, the system "learns" spatial configurations, allowing it to recall shapes even after the input is removed.</p>
                        </section>

                        <section>
                            <h3 className="text-lg font-bold text-white mb-2">Auto-Pilot Mode</h3>
                            <p className="leading-relaxed">Select the <strong>Auto-Pilot</strong> tab to run a pre-configured experiment. This automatically injects data, trains the network, wipes the memory, and then demonstrates associative recall.</p>
                        </section>

                        <section>
                            <h3 className="text-lg font-bold text-white mb-2">Manual Control</h3>
                            <ul className="list-disc pl-5 space-y-3">
                                <li>
                                    <strong className="text-cyan-300">1. Encode Data:</strong> Type a word (e.g., "BRAIN") and click INPUT. The particles will rearrange to form the shape.
                                </li>
                                <li>
                                    <strong className="text-pink-300">Target Regions:</strong> Use the buttons (Input A / Input B) to direct the input to specific parts of the cloud. This demonstrates how different sensory inputs can be processed separately.
                                </li>
                                <li>
                                    <strong className="text-yellow-300">2. Imprint (Learn):</strong> Toggle <strong>ON</strong>. The system calculates synaptic weights based on particle proximity. Active connections turn yellow. Wait for the structure to stabilize.
                                </li>
                                <li>
                                    <strong className="text-purple-300">3. Associative Memory:</strong> 
                                    <ul className="list-[circle] pl-5 mt-1 text-xs text-gray-400 space-y-1">
                                        <li><strong>Save:</strong> Stores the current weight matrix (synapses) into a slot.</li>
                                        <li><strong>Recall:</strong> Restores weights and quantum states (phase/spin) but <em>scrambles positions</em>. The system must then "hallucinate" the shape back into existence using the learned weights.</li>
                                    </ul>
                                </li>
                            </ul>
                        </section>
                    </div>
                </div>
                
                <div className="p-4 border-t border-gray-800 flex justify-end">
                     <button 
                        onClick={() => setShowHelp(false)}
                        className="px-4 py-2 bg-cyan-900/50 hover:bg-cyan-800 text-cyan-200 rounded border border-cyan-500/30 transition-colors"
                    >
                        Close Guide
                    </button>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

// --- Main App ---

const SimulationCanvas: React.FC<{ params: SimulationParams, dataRef: React.MutableRefObject<ParticleData> }> = ({ params, dataRef }) => {
  return (
    <Canvas camera={{ position: [0, 0, 35], fov: 50 }}>
        <color attach="background" args={['#020205']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="purple" />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <ParticleSystem params={params} dataRef={dataRef} />
        <OrbitControls autoRotate={false} />
    </Canvas>
  );
};

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  
  // Lift the data ref to the top level so both Canvas and UI can access it
  const dataRef = useRef<ParticleData>({
    x: new Float32Array(0),
    v: new Float32Array(0),
    phase: new Float32Array(0),
    spin: new Float32Array(0),
    activation: new Float32Array(0),
    target: new Float32Array(0),
    hasTarget: new Uint8Array(0),
    memoryMatrix: new Float32Array(0),
    regionID: new Uint8Array(0),
    forwardMatrix: new Float32Array(0),
    feedbackMatrix: new Float32Array(0),
    delayedActivation: new Float32Array(0),
    lastActiveTime: new Float32Array(0),
  });

  return (
    <div className="w-full h-screen bg-black overflow-hidden relative font-sans">
      <SimulationCanvas params={params} dataRef={dataRef} />
      <UIOverlay params={params} setParams={setParams} dataRef={dataRef} />
      
      <div className="absolute top-6 left-6 pointer-events-none max-w-lg hidden md:block">
        <h1 className="text-3xl font-bold text-white tracking-tight drop-shadow-md">
          Experiment: <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Free Energy Morphology</span>
        </h1>
        <div className="mt-2 bg-black/40 backdrop-blur-sm p-4 rounded-lg border-l-2 border-cyan-500">
           <p className="text-gray-300 text-sm leading-relaxed">
            "We examine whether a dynamic system of particles—characterized by spatial position, activation state, intrinsic spin, and vibrational phase—can facilitate self-supervised learning through free energy minimization."
           </p>
           <p className="text-gray-500 text-xs mt-2 font-mono">
             — Rawson, K. (2025). L-Group Predictive Coding Networks...
           </p>
        </div>
      </div>
    </div>
  );
}

export default App;