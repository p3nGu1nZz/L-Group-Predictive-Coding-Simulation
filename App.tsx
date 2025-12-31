import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { SimulationParams, ParticleData, DEFAULT_PARAMS, MemoryAction } from './types';

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
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  
  // Track system state for adaptive controls (prevents React render loops)
  const systemState = useRef({
      meanError: 10.0, // Start with high error assumption
      formationProgress: 0.0
  });

  const data = useRef<ParticleData>({
    x: new Float32Array(0),
    v: new Float32Array(0),
    phase: new Float32Array(0),
    spin: new Float32Array(0),
    activation: new Float32Array(0),
    target: new Float32Array(0),
    hasTarget: new Uint8Array(0),
    memoryMatrix: new Float32Array(0),
  });

  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    data.current = {
      x: new Float32Array(count * 3),
      v: new Float32Array(count * 3),
      phase: new Float32Array(count),
      spin: new Float32Array(count),
      activation: new Float32Array(count),
      target: new Float32Array(count * 3),
      hasTarget: new Uint8Array(count),
      memoryMatrix: new Float32Array(memorySize).fill(-1), 
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
    }
    
    memoryBank.current.clear();
    systemState.current.meanError = 10.0; 

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
                activation: new Float32Array(data.current.activation)
            };
            memoryBank.current.set(slot, snapshot);
            console.log(`Saved quantum state to Slot ${slot}`);
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

            // 3. DO NOT Restore Position (x) directly.
            // By NOT setting x, we force the network to "Infer" the state via Free Energy Minimization.
            // This aligns with the PCN framework where dynamics drive the system to the attractor.
            // data.current.x.set(snapshot.x); // <--- REMOVED
            
            // 4. Inject Thermal Noise (Entropy) to facilitate transition
            // This simulates "fuzzy moving" / destabilization of the previous state.
            for(let i=0; i<data.current.v.length; i++) {
                data.current.v[i] = (Math.random() - 0.5) * 0.5;
            }
            
            console.log(`Loaded quantum state from Slot ${slot} (Inference Initiated)`);
        }
    }
  }, [params.memoryAction]);

  useEffect(() => {
    if (params.memoryResetTrigger > 0) {
      if (data.current.memoryMatrix) {
        data.current.memoryMatrix.fill(-1);
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

        const activeCount = Math.min(count, targets.length);
        
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

        const pIndices = Array.from({length: count}, (_, i) => ({
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
  }, [params.inputText, params.particleCount]);

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

    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, spatialLearningRate, dataGravity, plasticity, damping } = params;
    const count = params.particleCount;
    const { x, v, phase, activation, target, hasTarget, memoryMatrix } = data.current;
    
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
        adaptiveStiffness = stiffness * 0.005; 
        const baseGravity = Math.max(dataGravity, 0.1);
        adaptiveGravity = baseGravity * (isLearning ? 6.0 : 3.0);
    } else {
        adaptiveStiffness = isLearning ? stiffness * 0.2 : stiffness;
        adaptiveGravity = isLearning ? Math.max(dataGravity, 0.6) : dataGravity;
    }

    // Adaptive Damping
    let effectiveDamping = 0.90; 

    if (hasActiveTargets) {
        effectiveDamping = isLearning ? 0.6 : 0.8;
    } else {
        const minRetention = 0.55; 
        const maxRetention = 0.98;
        const speedFactor = Math.exp(-meanVelocitySq * 1.5);
        effectiveDamping = minRetention + (maxRetention - minRetention) * speedFactor;
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

      for (let j = 0; j < count; j++) {
        if (i === j) continue;

        const jx = x[j * 3];
        const jy = x[j * 3 + 1];
        const jz = x[j * 3 + 2];

        const dx = jx - ix;
        const dy = jy - iy;
        const dz = jz - iz;
        const distSq = dx * dx + dy * dy + dz * dz;

        // --- PAPER IMPLEMENTATION: Eq 4.4 Probabilistic Coupling ---
        // p_ij(t) ~ exp(-d^2/sigma^2) * cos(phi_i - phi_j)
        // We use this probability to gate BOTH the force and the visual line.
        // This ensures that "messy" connections (close but out of phase) are suppressed.

        const dist = Math.sqrt(distSq);
        if (dist < 0.001 || dist > couplingDecay * 1.5) continue; // Optimization culling

        const phaseDiff = phase[j] - phase[i];
        
        // 1. Calculate Probabilistic Weight (Eq 4.4)
        // Normalized spatial decay
        const spatialWeight = Math.exp(-distSq / (couplingDecay * couplingDecay));
        
        // Phase alignment reward: (1 + cos) / 2 maps [-1, 1] to [0, 1]
        // This prevents negative probabilities while maintaining the "cosine reward" logic of the paper.
        const phaseWeight = (1.0 + Math.cos(phaseDiff)) * 0.5;
        
        const couplingProb = spatialWeight * phaseWeight;

        // 2. Memory & Learning Logic
        const memIndex = i * count + j;
        let r0 = memoryMatrix[memIndex];
        const isLearned = r0 !== -1;

        if (isLearning && couplingProb > 0.05) {
            // A. Structural Plasticity: Update equilibrium distance
            if (r0 === -1) r0 = dist; 
            r0 = r0 + (dist - r0) * plasticity;
            memoryMatrix[memIndex] = r0;

            // B. Phase Annealing: Sync phases of coupled particles to "burn in" the edge
            // This is critical for clean recall later.
            phaseDelta += Math.sin(phaseDiff) * plasticity * 2.0;
        }

        if (r0 === -1) r0 = effectiveEquilibrium;

        // 3. Force Calculation
        let localStiffness = adaptiveStiffness;
        if (dist < r0) localStiffness = Math.max(stiffness, 2.0); 

        // Modulate force by the probabilistic coupling. 
        // If they are out of phase, they shouldn't pull each other strongly.
        const effectiveForce = localStiffness * (dist - r0) * couplingProb;
        
        stress += Math.abs(effectiveForce);

        const nx = dx / dist;
        const ny = dy / dist;
        const nz = dz / dist;

        fx += nx * effectiveForce;
        fy += ny * effectiveForce;
        fz += nz * effectiveForce;

        // 4. Phase Synchronization (Kuramoto)
        if (couplingProb > 0.1) phaseDelta += couplingProb * Math.sin(phaseDiff);

        // 5. Visualization (Line Drawing)
        // We prioritize showing LEARNED bonds (memory) or very strong transient bonds.
        // This cleans up the "messy edges" by hiding weak probabilistic connections.
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
             // Learned bonds are Gold/Yellow
             r=1.0; g=0.84; b=0.0; 
             // Opacity based on how close they are to the learned equilibrium
             const err = Math.abs(dist - r0);
             alpha = Math.max(0.3, 1.0 - err); 
          } else {
             // Transient bonds: Dynamic Gradient
             const baseR = 0.1; const baseG = 0.2; const baseB = 0.6;
             const hotR = 0.8; const hotG = 0.9; const hotB = 1.0;
             
             r = baseR + (hotR - baseR) * excitation;
             g = baseG + (hotG - baseG) * excitation;
             b = baseB + (hotB - baseB) * excitation;
             
             // Opacity scales with coupling probability
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
             forceMag *= (distToTarget * 2.0); 
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

      const phaseColor = Math.sin(phase[i]) * 0.1;
      const r = 0.1 + energyLevel * 0.8;
      const g = 0.3 + energyLevel * 0.5 + phaseColor;
      const b = 0.8 + phaseColor;
      
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

// --- UI Overlay ---

interface UIOverlayProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams }) => {
  const [localText, setLocalText] = useState("");
  const [showInfo, setShowInfo] = useState(false);

  const handleChange = (key: keyof SimulationParams, value: number | string | object) => {
    setParams(prev => ({ ...prev, [key]: value }));
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

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end">
      <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded-xl p-5 text-white w-full shadow-2xl pointer-events-auto overflow-y-auto max-h-[90vh] custom-scrollbar">
        <h1 className="text-xl font-bold mb-1 text-cyan-400">Holographic Memory</h1>
        <p className="text-[10px] text-gray-400 mb-4 italic">Associative Storage in Particle Clouds</p>
        
        <div className="mb-4 bg-cyan-900/20 p-3 rounded-lg border border-cyan-500/30">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-cyan-300 mb-2">1. Encode Data</h2>
          <form onSubmit={handleTextSubmit} className="flex gap-2 mb-2">
            <input 
              type="text" 
              value={localText}
              onChange={(e) => setLocalText(e.target.value)}
              placeholder="e.g. QUBIT"
              className="w-full bg-black/50 border border-cyan-500/50 rounded px-2 py-1 text-sm focus:outline-none focus:border-cyan-400 font-mono"
              maxLength={16} // Increased from 8 to 16
            />
            <button type="submit" className="bg-cyan-600 hover:bg-cyan-500 text-white px-3 py-1 rounded text-xs font-bold transition-colors">
              INPUT
            </button>
          </form>
          
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
    </div>
  );
};

// --- Main App ---

const SimulationCanvas: React.FC<{ params: SimulationParams }> = ({ params }) => {
  return (
    <Canvas camera={{ position: [0, 0, 35], fov: 50 }}>
        <color attach="background" args={['#020205']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="purple" />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <ParticleSystem params={params} />
        <OrbitControls autoRotate={false} />
    </Canvas>
  );
};

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);

  return (
    <div className="w-full h-screen bg-black overflow-hidden relative font-sans">
      <SimulationCanvas params={params} />
      <UIOverlay params={params} setParams={setParams} />
      
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