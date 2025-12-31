import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Cylinder } from '@react-three/drei';
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

// Optimization: Singleton Canvas to avoid GC spikes on text change
let sharedTextCanvas: HTMLCanvasElement | null = null;
let sharedTextCtx: CanvasRenderingContext2D | null = null;

const textToPoints = (text: string): { positions: Float32Array, count: number } => {
  if (!text) return { positions: new Float32Array(0), count: 0 };

  if (!sharedTextCanvas) {
    sharedTextCanvas = document.createElement('canvas');
    sharedTextCanvas.width = 2048;
    sharedTextCanvas.height = 1024;
    sharedTextCtx = sharedTextCanvas.getContext('2d', { willReadFrequently: true });
  }

  const ctx = sharedTextCtx;
  if (!ctx) return { positions: new Float32Array(0), count: 0 };

  const width = sharedTextCanvas!.width;
  const height = sharedTextCanvas!.height;

  // Clear previous
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);
  
  const L = text.length;
  // Robust Inverse-Power Scaling
  const maxFontSize = 340;
  const minFontSize = 50;
  const fontSize = Math.max(minFontSize, maxFontSize / Math.pow(L, 0.45));

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

// Spatial Hash Grid Constants
const GRID_CELL_SIZE = 6.0; // Matched to cutoff distance approx
const GRID_DIM = 16;
const GRID_OFFSET = (GRID_DIM * GRID_CELL_SIZE) / 2; 
const TOTAL_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  // Grid Memory (Reuse to avoid GC)
  const gridRef = useRef({
      head: new Int32Array(TOTAL_CELLS).fill(-1),
      next: new Int32Array(params.particleCount).fill(-1)
  });

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  
  const systemState = useRef({
      meanError: 10.0, 
      logTimer: 0
  });

  const data = dataRef;

  // Initialization
  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    // Reallocate Grid if count changes
    if (gridRef.current.next.length !== count) {
        gridRef.current.next = new Int32Array(count).fill(-1);
    }
    
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

            // Region Init (Semantic Layering)
            // Top 25% = Input A (ID 0)
            // Mid 25% = Input B (ID 1)
            // Bot 50% = Assoc   (ID 2)
            if (i < Math.floor(count * 0.25)) data.current.regionID[i] = 0;      
            else if (i < Math.floor(count * 0.5)) data.current.regionID[i] = 1;  
            else data.current.regionID[i] = 2;                                   
        }
        
        memoryBank.current.clear();
        systemState.current.meanError = 10.0; 
    }
  }, [params.particleCount]);

  // Memory Operations
  useEffect(() => {
    const { type, slot } = params.memoryAction;
    if (type === 'idle') return;

    if (type === 'save') {
        const currentMatrix = data.current.memoryMatrix;
        if (currentMatrix) {
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
            console.log(`[Memory] Snapshot Saved to Slot ${slot}.`);
        }
    } else if (type === 'load') {
        const snapshot = memoryBank.current.get(slot);
        if (snapshot && data.current.memoryMatrix) {
            data.current.memoryMatrix.set(snapshot.matrix);
            data.current.target.set(snapshot.target);
            data.current.hasTarget.set(snapshot.hasTarget);
            
            data.current.phase.set(snapshot.phase);
            data.current.spin.set(snapshot.spin);
            data.current.activation.set(snapshot.activation);
            if(snapshot.regionID) data.current.regionID.set(snapshot.regionID);

            // Recall Optimization: Gentle Entropy Injection
            for(let i=0; i<data.current.v.length; i++) {
                data.current.v[i] = (Math.random() - 0.5) * 0.05; 
            }
            console.log(`[Memory] Loaded Slot ${slot}. Initiating Stabilized Recall.`);
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

  // Text / Target Processing
  useEffect(() => {
    const count = params.particleCount;
    const { positions, count: pointCount } = textToPoints(params.inputText);
    
    data.current.hasTarget.fill(0);

    if (pointCount > 0) {
        const currentPositions = data.current.x;
        const targets: {x: number, y: number, z: number}[] = [];
        
        if (pointCount > count) {
            const step = pointCount / count;
            for (let i = 0; i < count; i++) {
                const idx = Math.floor(i * step);
                targets.push({x: positions[idx * 3], y: positions[idx * 3 + 1], z: positions[idx * 3 + 2]});
            }
        } else {
            for (let i = 0; i < pointCount; i++) {
                targets.push({x: positions[i * 3], y: positions[i * 3 + 1], z: positions[i * 3 + 2]});
            }
        }
        
        const availableIndices = [];
        for (let i = 0; i < count; i++) {
            if (params.targetRegion === -1 || data.current.regionID[i] === params.targetRegion) {
                availableIndices.push(i);
            }
        }
        
        const activeCount = Math.min(availableIndices.length, targets.length);
        // Simple assignment for now, Morton optimization removed for brevity in this update
        for (let i = 0; i < activeCount; i++) {
            const pid = availableIndices[i];
            const t = targets[i];
            data.current.target[pid * 3] = t.x;
            data.current.target[pid * 3 + 1] = t.y;
            data.current.target[pid * 3 + 2] = t.z;
            data.current.hasTarget[pid] = 1;
            data.current.activation[pid] = 1.0;
        }
    }
    
    systemState.current.meanError = 10.0;

    if (ghostRef.current) {
      TEMP_OBJ.scale.set(0,0,0);
      for(let k=0; k<count; k++) ghostRef.current.setMatrixAt(k, TEMP_OBJ.matrix);
      for (let i = 0; i < count; i++) {
         if (data.current.hasTarget[i]) {
            TEMP_OBJ.position.set(data.current.target[i * 3], data.current.target[i * 3 + 1], data.current.target[i * 3 + 2]);
            TEMP_OBJ.scale.set(0.15, 0.15, 0.15);
            TEMP_OBJ.rotation.set(0,0,0);
            TEMP_OBJ.updateMatrix();
            ghostRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
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

  // PHYSICS LOOP
  useFrame((state) => {
    if (!meshRef.current || !linesRef.current || params.paused) return;

    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, spatialLearningRate, dataGravity, plasticity } = params;
    const count = params.particleCount;
    const { x, v, phase, activation, target, hasTarget, memoryMatrix, regionID } = data.current;
    
    if (x.length === 0) return;
    
    // --- 1. Update Spatial Grid ---
    const { head, next } = gridRef.current;
    head.fill(-1);
    next.fill(-1); // Reset only relevant range if optimization needed, but fill is fast

    for (let i = 0; i < count; i++) {
        const ix = Math.floor((x[i*3] + GRID_OFFSET) / GRID_CELL_SIZE);
        const iy = Math.floor((x[i*3+1] + GRID_OFFSET) / GRID_CELL_SIZE);
        const iz = Math.floor((x[i*3+2] + GRID_OFFSET) / GRID_CELL_SIZE);
        
        if (ix >= 0 && ix < GRID_DIM && iy >= 0 && iy < GRID_DIM && iz >= 0 && iz < GRID_DIM) {
            const cellIdx = ix + iy * GRID_DIM + iz * GRID_DIM * GRID_DIM;
            next[i] = head[cellIdx];
            head[cellIdx] = i;
        }
    }

    let activeTargetCount = 0;
    for(let k = 0; k < count; k++) if(hasTarget[k] === 1) activeTargetCount++;
    const hasActiveTargets = activeTargetCount > 0;

    // --- Pre-Calculate Global Stats ---
    let totalActivation = 0, totalRadiusSq = 0, totalVelocitySq = 0;
    for (let k = 0; k < count; k++) {
      totalActivation += activation[k];
      const px = x[k * 3], py = x[k * 3 + 1], pz = x[k * 3 + 2];
      totalRadiusSq += px * px + py * py + pz * pz;
      const vx = v[k * 3], vy = v[k * 3 + 1], vz = v[k * 3 + 2];
      totalVelocitySq += vx * vx + vy * vy + vz * vz;
    }
    const meanActivation = count > 0 ? totalActivation / count : 0;
    const meanRadius = count > 0 ? Math.sqrt(totalRadiusSq / count) : 8.0;
    const meanVelocitySq = count > 0 ? totalVelocitySq / count : 0;
    
    // Logging
    systemState.current.logTimer += 1;
    if (systemState.current.logTimer > 120 && params.memoryAction.type === 'idle' && !hasActiveTargets && meanVelocitySq > 0.01) {
         console.debug(`[Physics] KE: ${meanVelocitySq.toFixed(5)}`);
         systemState.current.logTimer = 0;
    }

    // Adaptive Parameters
    const currentError = systemState.current.meanError;
    let adaptiveRateScale = hasActiveTargets 
        ? (currentError > 4.0 ? 1.2 : (currentError > 0.8 ? 0.3 : 0.05 + currentError * 0.3))
        : 1.0 / (1.0 + meanVelocitySq * 0.2);
    
    const effectiveLearningRate = spatialLearningRate * adaptiveRateScale;

    // Density Equilibrium
    const currentVol = Math.max(1.0, Math.pow(meanRadius, 3));
    const nominalVol = Math.pow(8.0, 3);
    const densityScale = 1.0 / Math.pow((count / currentVol) / (800.0 / nominalVol), 0.333);
    const effectiveEquilibrium = equilibriumDistance * Math.max(0.5, Math.min(2.0, densityScale));

    const isLearning = plasticity > 0;
    let adaptiveStiffness = stiffness;
    let adaptiveGravity = dataGravity;

    if (hasActiveTargets) {
        adaptiveStiffness = stiffness * 0.005; 
        adaptiveGravity = Math.max(dataGravity, 0.1) * (isLearning ? 12.0 : 6.0); 
    } else {
        adaptiveStiffness = isLearning ? stiffness * 0.2 : stiffness;
        adaptiveGravity = isLearning ? Math.max(dataGravity, 0.6) : dataGravity;
    }

    let effectiveDamping = params.damping;
    if (hasActiveTargets) {
        effectiveDamping = isLearning ? 0.5 : 0.8; 
    } else {
        effectiveDamping = 0.55; 
    }

    const systemStress = Math.min(1.0, meanActivation);
    const excitation = Math.min(1.0, systemStress * 0.5 + Math.min(1.0, meanVelocitySq * 0.5) * 0.5);
    const connectionThreshold = 0.15 + (0.45) * (excitation * excitation);

    const decaySq = couplingDecay * couplingDecay;
    const invDecaySq = 1.0 / decaySq;
    const cutoffDist = couplingDecay * 1.5;

    let lineIndex = 0;
    const linePositions = linesRef.current.geometry.attributes.position.array as Float32Array;
    const lineColors = linesRef.current.geometry.attributes.color.array as Float32Array;
    let frameTotalDist = 0;

    // --- Main Loop (Spatial Hash Optimized) ---
    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0;
      let stress = 0;

      const ix = x[i * 3], iy = x[i * 3 + 1], iz = x[i * 3 + 2];
      const ri = regionID[i];
      
      const gridX = Math.floor((ix + GRID_OFFSET) / GRID_CELL_SIZE);
      const gridY = Math.floor((iy + GRID_OFFSET) / GRID_CELL_SIZE);
      const gridZ = Math.floor((iz + GRID_OFFSET) / GRID_CELL_SIZE);

      // Iterate Neighbor Cells (3x3x3)
      for (let gx = gridX - 1; gx <= gridX + 1; gx++) {
          if (gx < 0 || gx >= GRID_DIM) continue;
          for (let gy = gridY - 1; gy <= gridY + 1; gy++) {
             if (gy < 0 || gy >= GRID_DIM) continue;
             for (let gz = gridZ - 1; gz <= gridZ + 1; gz++) {
                 if (gz < 0 || gz >= GRID_DIM) continue;
                 
                 const cellIdx = gx + gy * GRID_DIM + gz * GRID_DIM * GRID_DIM;
                 let j = head[cellIdx];
                 
                 while (j !== -1) {
                    if (i !== j) {
                        const rj = regionID[j];
                        
                        // Inhibition Rule: Input A (0) cannot talk directly to Input B (1)
                        const allowed = !((ri === 0 && rj === 1) || (ri === 1 && rj === 0));
                        
                        if (allowed) {
                            const dx = x[j * 3] - ix;
                            const dy = x[j * 3 + 1] - iy;
                            const dz = x[j * 3 + 2] - iz;
                            const distSq = dx * dx + dy * dy + dz * dz;

                            if (distSq > 0.0001 && distSq < cutoffDist * cutoffDist) {
                                const dist = Math.sqrt(distSq);
                                
                                const phaseDiff = phase[j] - phase[i];
                                const spatialWeight = Math.exp(-distSq * invDecaySq);
                                const couplingProb = spatialWeight * (0.5 + 0.5 * Math.cos(phaseDiff));

                                const memIndex = i * count + j;
                                let r0 = memoryMatrix[memIndex];
                                const isLearned = r0 !== -1;

                                if (isLearning && couplingProb > 0.05) {
                                    if (r0 === -1) r0 = dist; 
                                    if (meanVelocitySq < 0.05 || r0 === dist) {
                                        r0 = r0 + (dist - r0) * plasticity;
                                    }
                                    memoryMatrix[memIndex] = r0;
                                    phaseDelta += Math.sin(phaseDiff) * plasticity * 2.0;
                                }

                                if (r0 === -1) r0 = effectiveEquilibrium;

                                let localStiffness = adaptiveStiffness;
                                if (dist < r0) localStiffness = Math.max(stiffness, 2.0); 
                                if (ri === rj && (ri === 0 || ri === 1)) localStiffness *= 5.0; 
                                
                                if (!hasActiveTargets && isLearned) localStiffness *= 8.0; 

                                const effectiveForce = localStiffness * (dist - r0) * couplingProb;
                                stress += Math.abs(effectiveForce);

                                const invDist = 1.0 / dist;
                                fx += dx * invDist * effectiveForce;
                                fy += dy * invDist * effectiveForce;
                                fz += dz * invDist * effectiveForce;

                                if (couplingProb > 0.1) phaseDelta += couplingProb * Math.sin(phaseDiff);

                                // Visualization (limited count)
                                const showLine = isLearned || (couplingProb > connectionThreshold);
                                if (j > i && showLine && lineIndex < maxConnections) {
                                    const idx6 = lineIndex * 6;
                                    linePositions[idx6] = ix; linePositions[idx6 + 1] = iy; linePositions[idx6 + 2] = iz;
                                    linePositions[idx6 + 3] = x[j*3]; linePositions[idx6 + 4] = x[j*3+1]; linePositions[idx6 + 5] = x[j*3+2];

                                    let r, g, b, alpha;
                                    if (isLearned) {
                                        r=1.0; g=0.84; b=0.0; 
                                        alpha = Math.max(0.3, 1.0 - Math.abs(dist - r0)); 
                                    } else {
                                        r = 0.1 + 0.7 * excitation;
                                        g = 0.2 + 0.7 * excitation;
                                        b = 0.6 + 0.4 * excitation;
                                        alpha = Math.sqrt(Math.max(0, couplingProb - connectionThreshold) / (1.0 - connectionThreshold));
                                    }
                                    lineColors[idx6] = r * alpha; lineColors[idx6 + 1] = g * alpha; lineColors[idx6 + 2] = b * alpha;
                                    lineColors[idx6 + 3] = r * alpha; lineColors[idx6 + 4] = g * alpha; lineColors[idx6 + 5] = b * alpha;
                                    lineIndex++;
                                }
                            }
                        }
                    }
                    j = next[j];
                 }
             }
          }
      }

      // Target Gravity
      if (hasTarget[i]) {
        const dx = target[i * 3] - ix;
        const dy = target[i * 3 + 1] - iy;
        const dz = target[i * 3 + 2] - iz;
        const distToTarget = Math.sqrt(dx*dx + dy*dy + dz*dz);
        frameTotalDist += distToTarget;

        let forceMag = adaptiveGravity; 
        if (distToTarget > 0.5) {
            forceMag *= (1.0 + Math.pow(distToTarget, 0.6));
        } else {
             forceMag *= (distToTarget * 8.0); 
        }
        
        const invDist = 1.0 / (distToTarget + 0.0001);
        fx += dx * invDist * forceMag;
        fy += dy * invDist * forceMag;
        fz += dz * invDist * forceMag;
        
        const idx3 = i*3;
        const speedSq = v[idx3]*v[idx3] + v[idx3+1]*v[idx3+1] + v[idx3+2]*v[idx3+2];
        if (speedSq > 1.0) {
            const scale = 1.0 / Math.sqrt(speedSq);
            v[idx3] *= scale; v[idx3+1] *= scale; v[idx3+2] *= scale;
        }
      }

      activation[i] = activation[i] * 0.95 + stress * 0.05;

      const idx3 = i * 3;
      v[idx3] = v[idx3] * effectiveDamping + fx * effectiveLearningRate;
      v[idx3 + 1] = v[idx3 + 1] * effectiveDamping + fy * effectiveLearningRate;
      v[idx3 + 2] = v[idx3 + 2] * effectiveDamping + fz * effectiveLearningRate;

      x[idx3] += v[idx3];
      x[idx3 + 1] += v[idx3 + 1];
      x[idx3 + 2] += v[idx3 + 2];

      const rSq = x[idx3]*x[idx3] + x[idx3 + 1]*x[idx3 + 1] + x[idx3 + 2]*x[idx3 + 2];
      if (rSq > 900) {
           const scale = 0.98;
           x[idx3] *= scale; x[idx3 + 1] *= scale; x[idx3 + 2] *= scale;
      }

      phase[i] += phaseSyncRate * phaseDelta + 0.02;

      TEMP_OBJ.position.set(x[idx3], x[idx3 + 1], x[idx3 + 2]);
      const energyLevel = Math.min(1.0, activation[i]); 
      const s = 0.4 + 0.4 * energyLevel;
      TEMP_OBJ.scale.set(s, s, s);
      TEMP_OBJ.updateMatrix();
      meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      const r_id = regionID[i];
      let br=0, bg=0, bb=0;
      if (r_id === 0) { br=0.0; bg=1.0; bb=1.0; }
      else if (r_id === 1) { br=1.0; bg=0.4; bb=0.8; }
      else { br=1.0; bg=0.84; bb=0.0; }
      
      const pm = Math.sin(phase[i]) * 0.1;
      
      let stabilityBoost = 0;
      if (!hasActiveTargets) {
          const speed = v[idx3]*v[idx3] + v[idx3+1]*v[idx3+1] + v[idx3+2]*v[idx3+2];
          if (speed < 0.005) stabilityBoost = 0.5;
      }

      TEMP_COLOR.setRGB(
          Math.min(1, br + energyLevel*0.4 + pm + stabilityBoost), 
          Math.min(1, bg + energyLevel*0.4 + pm + stabilityBoost), 
          Math.min(1, bb + energyLevel*0.4 + pm + stabilityBoost)
      );
      meshRef.current.setColorAt(i, TEMP_COLOR);
    }
    
    systemState.current.meanError = hasActiveTargets ? frameTotalDist / activeTargetCount : 0;

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

// --- Region Guides Component ---
const RegionGuides: React.FC<{ params: SimulationParams }> = ({ params }) => {
    if (!params.showRegions) return null;
    
    // Y-ranges based on Initialization Logic (Radius ~14, Linear Y distribution)
    // Input A: Top 25% -> Y: 7 to 14
    // Input B: Mid 25% -> Y: 0 to 7
    // Assoc: Bot 50%   -> Y: -14 to 0
    
    return (
        <group>
            {/* Input A - Cyan - Top */}
            <Cylinder args={[14, 14, 7, 32]} position={[0, 10.5, 0]}>
                <meshBasicMaterial color="cyan" wireframe transparent opacity={0.15} />
            </Cylinder>
            <group position={[15, 10.5, 0]}>
                 {/* Label logic could go here, for now using just geometry */}
            </group>

            {/* Input B - Pink - Mid */}
            <Cylinder args={[14, 14, 7, 32]} position={[0, 3.5, 0]}>
                <meshBasicMaterial color="#ff66cc" wireframe transparent opacity={0.15} />
            </Cylinder>

            {/* Associative - Gold - Bottom */}
            <Cylinder args={[14, 14, 14, 32]} position={[0, -7, 0]}>
                <meshBasicMaterial color="#ffd700" wireframe transparent opacity={0.15} />
            </Cylinder>
        </group>
    );
};

// --- Memory Matrix Components ---

interface MemoryMatrixProps {
    dataRef: React.MutableRefObject<ParticleData>;
    params: SimulationParams;
    size?: number;
    className?: string;
    onClick?: () => void;
    label?: boolean;
}

const MemoryMatrix: React.FC<MemoryMatrixProps> = ({ dataRef, params, size = 256, className, onClick, label = true }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
        let animationFrameId: number;
        const render = () => {
            if (!canvasRef.current || !dataRef.current.memoryMatrix) return;
            const count = params.particleCount;
            const ctx = canvasRef.current.getContext('2d');
            if (!ctx) return;
            
            if (canvasRef.current.width !== size) {
                canvasRef.current.width = size;
                canvasRef.current.height = size;
            }

            const matrix = dataRef.current.memoryMatrix;
            const imgData = ctx.createImageData(size, size);
            const pixels = new Uint32Array(imgData.data.buffer);
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const idx = (y * size + x);
                    const matIdx = y * count + x; 
                    if (y < count && x < count) {
                         const val = matrix[matIdx];
                         pixels[idx] = (val !== -1) ? 0xFF00D7FF : 0xFF101010; // Gold or Dark Gray
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
    }, [params.particleCount, size]);

    return (
        <div onClick={onClick} className={`relative group cursor-pointer ${className}`}>
             <canvas ref={canvasRef} className="w-full h-full image-pixelated rounded border border-cyan-500/30 group-hover:border-cyan-400 transition-colors bg-black" />
             {label && (
                 <div className="absolute bottom-1 left-1 right-1 bg-black/60 backdrop-blur px-2 py-1 flex justify-between items-center rounded pointer-events-none">
                    <span className="text-[9px] text-cyan-400 font-bold uppercase tracking-wider">Synaptic Matrix</span>
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-3 h-3 text-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                    </svg>
                 </div>
             )}
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
  const [showHelp, setShowHelp] = useState(false);
  const [activeTab, setActiveTab] = useState<'auto' | 'manual'>('auto');
  
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

  useEffect(() => {
    if (autoRunPhase === 'idle') return;
    let timer: ReturnType<typeof setTimeout>;

    if (autoRunPhase === 'reset') {
        setAutoStatus("Reset System");
        setParams(DEFAULT_PARAMS);
        setLocalText("");
        timer = setTimeout(() => setAutoRunPhase('inject'), 1500);
    } 
    else if (autoRunPhase === 'inject') {
        const word = "Q-MIND";
        setAutoStatus(`Inject "${word}"`);
        setLocalText(word);
        setParams(prev => ({ ...prev, inputText: word, dataGravity: 0.4 }));
        timer = setTimeout(() => setAutoRunPhase('stabilize'), 3000);
    }
    else if (autoRunPhase === 'stabilize') {
        setAutoStatus("Stabilizing...");
        timer = setTimeout(() => setAutoRunPhase('learning'), 3500);
    }
    else if (autoRunPhase === 'learning') {
        setAutoStatus("Hebbian Learning");
        togglePlasticity(true);
        timer = setTimeout(() => setAutoRunPhase('saving'), 5000);
    }
    else if (autoRunPhase === 'saving') {
        setAutoStatus("Persisting...");
        handleMemoryAction('save', 1);
        timer = setTimeout(() => setAutoRunPhase('forgetting'), 1500);
    }
    else if (autoRunPhase === 'forgetting') {
        setAutoStatus("Forgetting...");
        togglePlasticity(false);
        setLocalText("");
        setParams(prev => ({ ...prev, inputText: "", memoryResetTrigger: prev.memoryResetTrigger + 1 }));
        timer = setTimeout(() => setAutoRunPhase('recalling'), 3000);
    }
    else if (autoRunPhase === 'recalling') {
        setAutoStatus("Recalling...");
        handleMemoryAction('load', 1);
        timer = setTimeout(() => {
            setAutoStatus("Complete");
            setAutoRunPhase('idle');
        }, 5000);
    }
    return () => clearTimeout(timer);
  }, [autoRunPhase]);

  const startAutoRun = () => setAutoRunPhase('reset');

  return (
    <div className="absolute top-0 right-0 p-3 w-full md:w-72 h-full pointer-events-none flex flex-col items-end">
      <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded-xl p-3 text-white w-full shadow-2xl pointer-events-auto flex flex-col gap-2">
        
        <div className="flex justify-between items-center">
            <h1 className="text-sm font-bold text-cyan-400">Holographic Memory</h1>
            <div className="flex items-center gap-2">
                 <button 
                    onClick={togglePause}
                    className={`p-1 rounded hover:bg-white/10 ${params.paused ? 'text-yellow-400' : 'text-gray-400'}`}
                 >
                    {params.paused ? (
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" /></svg>
                    ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" /></svg>
                    )}
                 </button>
                 <button onClick={() => setShowHelp(true)} className="text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12V8.25z" /></svg>
                </button>
            </div>
        </div>

        {/* Compact Tab Switcher */}
        <div className="flex bg-gray-900 rounded p-0.5 border border-gray-700">
            <button 
                onClick={() => setActiveTab('auto')}
                className={`flex-1 py-1 text-[10px] font-bold rounded ${activeTab === 'auto' ? 'bg-cyan-600 text-white' : 'text-gray-500 hover:text-white'}`}
            >
                Auto-Pilot
            </button>
            <button 
                onClick={() => setActiveTab('manual')}
                className={`flex-1 py-1 text-[10px] font-bold rounded ${activeTab === 'manual' ? 'bg-purple-600 text-white' : 'text-gray-500 hover:text-white'}`}
            >
                Manual
            </button>
        </div>

        {/* AUTO VIEW */}
        {activeTab === 'auto' && (
            <div className="space-y-2">
                <div className="bg-cyan-900/10 border border-cyan-500/30 rounded p-2 text-center">
                    {autoRunPhase === 'idle' ? (
                        <button 
                            onClick={startAutoRun}
                            className="w-full py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded transition-colors"
                        >
                            Start Experiment
                        </button>
                    ) : (
                        <div className="text-left bg-black/40 p-2 rounded">
                             <div className="flex items-center gap-2 mb-1">
                                <span className="relative flex h-2 w-2">
                                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                                </span>
                                <span className="text-[10px] text-cyan-300 font-bold uppercase">Running</span>
                             </div>
                             <p className="text-[10px] text-white font-mono">{autoStatus}</p>
                        </div>
                    )}
                </div>
            </div>
        )}
        
        {/* MANUAL VIEW */}
        {activeTab === 'manual' && (
            <div className="space-y-2">
                <div className="bg-cyan-900/20 p-2 rounded border border-cyan-500/30">
                  <div className="flex justify-between items-center mb-1">
                      <h2 className="text-[9px] font-bold text-cyan-300 uppercase">Input</h2>
                      <button onClick={clearInput} className="text-[9px] text-red-300 hover:text-red-100 uppercase">Clear</button>
                  </div>
                  <form onSubmit={handleTextSubmit} className="flex gap-1 mb-2">
                    <input 
                      type="text" 
                      value={localText}
                      onChange={(e) => setLocalText(e.target.value)}
                      className="w-full bg-black/50 border border-cyan-500/50 rounded px-1.5 py-0.5 text-xs text-cyan-100 font-mono"
                      maxLength={10} 
                    />
                    <button type="submit" className="bg-cyan-600 text-white px-2 rounded text-[10px] font-bold">GO</button>
                  </form>
                  
                  <div className="flex gap-1">
                     {[0, 1, -1].map(r => (
                         <button 
                             key={r}
                             onClick={() => handleChange('targetRegion', r)}
                             className={`flex-1 text-[8px] py-0.5 rounded border ${params.targetRegion === r ? 'bg-cyan-500 text-black font-bold border-transparent' : 'border-gray-700 text-gray-400'}`}
                         >
                             {r === 0 ? 'In A' : r === 1 ? 'In B' : 'All'}
                         </button>
                     ))}
                  </div>
                </div>

                <div className="bg-yellow-900/20 p-2 rounded border border-yellow-500/30 flex items-center justify-between">
                    <div>
                        <div className="text-[9px] font-bold text-yellow-300 uppercase">Plasticity</div>
                        <div className="text-[8px] text-yellow-500/70">{params.plasticity > 0 ? "Hebbian ON" : "Fixed Weights"}</div>
                    </div>
                    <button 
                        onClick={() => togglePlasticity(params.plasticity === 0)}
                        className={`px-3 py-1 rounded text-[9px] font-bold ${params.plasticity > 0 ? 'bg-yellow-500 text-black' : 'bg-gray-700 text-gray-400'}`}
                    >
                        {params.plasticity > 0 ? 'ACTIVE' : 'OFF'}
                    </button>
                </div>
                
                 {/* Visualization Toggle */}
                <div className="bg-gray-800/40 p-2 rounded border border-gray-700 flex items-center justify-between">
                     <span className="text-[9px] font-bold text-gray-300 uppercase">Regions</span>
                     <button 
                        onClick={() => handleChange('showRegions', !params.showRegions)}
                        className={`w-8 h-4 rounded-full relative transition-colors ${params.showRegions ? 'bg-cyan-600' : 'bg-gray-600'}`}
                     >
                        <span className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${params.showRegions ? 'left-4.5' : 'left-0.5'}`} style={{ left: params.showRegions ? '18px' : '2px' }} />
                     </button>
                </div>

                <div className="bg-purple-900/20 p-2 rounded border border-purple-500/30">
                     <div className="flex justify-between items-center mb-1">
                        <h2 className="text-[9px] font-bold text-purple-300 uppercase">Memory</h2>
                        <button onClick={forgetCurrentMemory} className="text-[8px] text-gray-400 hover:text-white uppercase">Wipe</button>
                     </div>
                     <div className="grid grid-cols-4 gap-1">
                        {[1, 2, 3, 4].map(slot => (
                            <div key={slot} className="flex flex-col gap-0.5">
                                <button onClick={() => handleMemoryAction('load', slot)} className="bg-purple-600 hover:bg-purple-500 text-white text-[9px] py-1 rounded">R{slot}</button>
                                <button onClick={() => handleMemoryAction('save', slot)} className="bg-purple-900/50 hover:bg-purple-800 text-purple-300 text-[8px] py-0.5 rounded border border-purple-500/30">Save</button>
                            </div>
                        ))}
                     </div>
                </div>
            </div>
        )}

        {/* Compact Legend */}
        <div className="flex justify-between text-[8px] text-gray-500 px-1 mt-1">
             <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span> In A</div>
             <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-pink-400"></span> In B</div>
             <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-yellow-400"></span> Assoc</div>
        </div>

        <button 
          onClick={reset}
          className="w-full py-1 mt-1 bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-500/20 rounded text-[9px] uppercase tracking-wider"
        >
          Reset Simulation
        </button>
      </div>
      
      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 pointer-events-auto">
            <div className="bg-gray-900 border border-cyan-500/50 rounded-xl w-full max-w-lg p-4 relative shadow-2xl">
                <button onClick={() => setShowHelp(false)} className="absolute top-3 right-3 text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
                <h2 className="text-lg font-bold text-cyan-400 mb-2">Guide</h2>
                <div className="space-y-2 text-xs text-gray-300">
                    <p><strong>Auto-Pilot:</strong> Runs a full training cycle automatically.</p>
                    <p><strong>Manual:</strong></p>
                    <ul className="list-disc pl-4 space-y-1">
                        <li>Type text + GO to form shapes.</li>
                        <li>Turn <strong>Plasticity ON</strong> to learn the shape (Wait for yellow connections).</li>
                        <li><strong>Save</strong> to a slot.</li>
                        <li><strong>Wipe</strong> to clear the cloud.</li>
                        <li><strong>R1-R4</strong> to recall the shape from memory (hallucination).</li>
                        <li><strong>Regions:</strong> Toggle visualization of the semantic layers.</li>
                    </ul>
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
        <RegionGuides params={params} />
        <OrbitControls autoRotate={false} />
    </Canvas>
  );
};

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  const [matrixModalOpen, setMatrixModalOpen] = useState(false);
  
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
      
      {/* Heads-Up Display: Memory Matrix (Bottom Left) */}
      <div className="absolute bottom-6 left-6 flex flex-col gap-2 pointer-events-auto">
          <MemoryMatrix 
             dataRef={dataRef} 
             params={params} 
             size={256} 
             className="w-40 h-40 shadow-2xl" 
             onClick={() => setMatrixModalOpen(true)}
          />
      </div>

      <UIOverlay params={params} setParams={setParams} dataRef={dataRef} />
      
      {/* Title (Hidden on small screens, absolute top left) */}
      <div className="absolute top-6 left-6 pointer-events-none max-w-lg hidden md:block opacity-50 hover:opacity-100 transition-opacity">
        <h1 className="text-2xl font-bold text-white tracking-tight">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Free Energy Morphology</span>
        </h1>
        <p className="text-[10px] text-gray-500 font-mono mt-1">v0.9.3 // L-Group Predictive Coding</p>
      </div>

      {/* Large Matrix Modal */}
      {matrixModalOpen && (
          <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-8 animate-in fade-in duration-200" onClick={() => setMatrixModalOpen(false)}>
              <div className="relative w-full max-w-3xl aspect-square flex flex-col items-center justify-center" onClick={(e) => e.stopPropagation()}>
                   <button onClick={() => setMatrixModalOpen(false)} className="absolute -top-10 right-0 text-white hover:text-cyan-400 uppercase text-xs font-bold tracking-widest">
                       Close Analysis
                   </button>
                   <MemoryMatrix 
                       dataRef={dataRef} 
                       params={params} 
                       size={1024} 
                       className="w-full h-full shadow-2xl border-2 border-cyan-500/20 rounded-lg cursor-default" 
                       label={false}
                    />
                    <div className="absolute bottom-4 left-4 text-xs text-cyan-500 font-mono bg-black/80 px-3 py-1 rounded border border-cyan-900">
                        Matrix Resolution: 1024x1024 | Synaptic Density Visualization
                    </div>
              </div>
          </div>
      )}
    </div>
  );
}

export default App;