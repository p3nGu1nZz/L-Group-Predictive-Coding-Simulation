import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Cylinder, Grid } from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Noise, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { SimulationParams, ParticleData, DEFAULT_PARAMS, MemoryAction, CONSTANTS } from './types';

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

// Optimization: Singleton Canvas
let sharedTextCanvas: HTMLCanvasElement | null = null;
let sharedTextCtx: CanvasRenderingContext2D | null = null;

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

interface SystemStats {
  meanError: number;
  meanSpeed: number;
  energy: number;
  fps: number;
}

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
}

interface MemorySnapshot {
  x: Float32Array;
  regionID: Uint8Array;
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef, statsRef, started, spatialRefs }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const outlineRef = useRef<THREE.InstancedMesh>(null); 
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  const data = dataRef;

  // Initialization
  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    spatialRefs.current.gridNext = new Int32Array(count);
    spatialRefs.current.neighborList = new Int32Array(count * 128); 
    spatialRefs.current.neighborCounts = new Int32Array(count);

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
            
            if (i < Math.floor(count * 0.25)) data.current.regionID[i] = 0;      
            else if (i < Math.floor(count * 0.5)) data.current.regionID[i] = 1;  
            else data.current.regionID[i] = 2;                                   
        }
        memoryBank.current.clear();
    }
  }, [params.particleCount]);

  // Memory Action Handler (Save/Load/Crosstalk)
  useEffect(() => {
    const { type, slot, triggerId } = params.memoryAction;
    if (type === 'idle' || triggerId === 0) return;

    if (type === 'save') {
        const snapshot: MemorySnapshot = {
            x: new Float32Array(data.current.x),
            regionID: new Uint8Array(data.current.regionID)
        };
        memoryBank.current.set(slot, snapshot);
        console.log(`[MEMORY] Saved state to Slot ${slot}`);
    } 
    else if (type === 'load') {
        // CROSSTALK STRESS TEST (Sprint 0 Requirement)
        // Load Slot -1 means "Merge ALL memories" to test capacity
        if (slot === -1) {
            const keys = Array.from(memoryBank.current.keys());
            if (keys.length > 0) {
                 console.log(`[MEMORY] SIMULATING CROSSTALK: Merging ${keys.length} slots`);
                 
                 // Clear current targets
                 data.current.target.fill(0);
                 
                 // Superposition: Sum all positions
                 for (const k of keys) {
                     const snap = memoryBank.current.get(k);
                     if (snap) {
                         for(let i=0; i<data.current.target.length; i++) {
                             data.current.target[i] += snap.x[i];
                         }
                     }
                 }

                 // Normalize (Average)
                 const invLen = 1.0 / keys.length;
                 for(let i=0; i<data.current.target.length; i++) {
                     data.current.target[i] *= invLen;
                 }
                 
                 data.current.hasTarget.fill(1);
                 data.current.v.fill(0); // Kill momentum
            } else {
                console.warn("[MEMORY] No patterns saved for Crosstalk test.");
            }
        }
        else if (slot === -2) {
             // Clear Memory Bank Command
             memoryBank.current.clear();
             console.log("[MEMORY] Bank Cleared");
        }
        else {
            const snapshot = memoryBank.current.get(slot);
            if (snapshot) {
                data.current.target.set(snapshot.x);
                data.current.hasTarget.fill(1);
                data.current.v.fill(0);
                console.log(`[MEMORY] Recalled state from Slot ${slot}`);
            }
        }
    }
  }, [params.memoryAction.triggerId]);

  // Text Processing
  useEffect(() => {
    if (params.paused) return; 
    
    // In stress test mode, we might switch inputs rapidly.
    // If text is empty, clear targets.
    if (!params.inputText) {
        data.current.hasTarget.fill(0);
        return;
    }

    const count = params.particleCount;
    // Utilize more particles (95% instead of 80%) to increase density
    const { positions, count: pointCount } = textToPoints(params.inputText, Math.floor(count * 0.95));
    
    data.current.hasTarget.fill(0);

    if (pointCount > 0) {
        const targets = [];
        for(let i=0; i<pointCount; i++) {
            targets.push({ x: positions[i*3], y: positions[i*3+1], z: positions[i*3+2] });
        }
        
        // FIX: Shuffle targets array to prevent bottom cropping.
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
            const t = targets[i];
            
            if (params.targetRegion !== -1 && data.current.regionID[pid] !== params.targetRegion) {
                continue;
            }

            data.current.target[pid * 3] = t.x;
            data.current.target[pid * 3 + 1] = t.y;
            data.current.target[pid * 3 + 2] = t.z;
            data.current.hasTarget[pid] = 1;
            data.current.activation[pid] = 1.0;
        }
    }
  }, [params.inputText, params.particleCount, params.targetRegion, params.paused]);

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
    
    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, plasticity } = params;
    const count = params.particleCount;
    const { x, v, phase, target, hasTarget, regionID } = data.current;
    
    if (x.length === 0) return;

    let activeTargetCount = 0;
    for(let k = 0; k < count; k++) if(hasTarget[k] === 1) activeTargetCount++;
    const hasActiveTargets = activeTargetCount > 0;
    
    const isEncoding = plasticity > 0;
    const isRecalling = hasActiveTargets && params.inputText === ""; 

    const k_spring_base = isEncoding ? 0.2 : (isRecalling ? 1.5 : (started ? 0.2 : 0.05));
    const stiffnessMult = isRecalling ? 0.1 : (isEncoding ? 0.1 : 1.0);

    // Spatial Hashing
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
                                 if (distSq < 25.0) { 
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
    let totalKineticEnergy = 0;

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0;
      const idx3 = i * 3;
      const ix = x[idx3], iy = x[idx3 + 1], iz = x[idx3 + 2];
      const rid = regionID[i];
      const nOffset = i * 48; 
      const nCount = spatialRefs.current.neighborCounts[i];
      const isTarget = hasTarget[i] === 1;

      if (effectiveChaos) {
          fx += (Math.random() - 0.5) * 1.5;
          fy += (Math.random() - 0.5) * 1.5;
          fz += (Math.random() - 0.5) * 1.5;
          fx += -iy * 0.05; fy += ix * 0.05;
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
          
          fx += dx * k_spring_base;
          fy += dy * k_spring_base;
          fz += dz * k_spring_base;
      }

      const interactionStrength = hasTarget[i] ? 0.1 : 1.0;
      if (interactionStrength > 0.01) {
          for (let n = 0; n < nCount; n++) {
            const j = spatialRefs.current.neighborList[nOffset + n];
            const rj = regionID[j];
            
            if ((rid === 0 && rj === 1) || (rid === 1 && rj === 0)) continue;

            const dx = x[j*3] - ix; const dy = x[j*3+1] - iy; const dz = x[j*3+2] - iz;
            const distSq = dx*dx + dy*dy + dz*dz;
            if (distSq < 0.01 || distSq > 16.0) continue; 
            const dist = Math.sqrt(distSq);
            const r0 = equilibriumDistance; 
            let force = 0;
            
            if (dist < r0) force = -stiffness * stiffnessMult * (r0 - dist) * 2.0;
            else if (!hasTarget[i]) force = stiffness * stiffnessMult * (dist - r0) * 0.1;
            
            force *= interactionStrength;
            const invDist = 1.0 / dist;
            fx += dx * invDist * force; fy += dy * invDist * force; fz += dz * invDist * force;
            const phaseDiff = phase[j] - phase[i];
            phaseDelta += Math.sin(phaseDiff) * 0.1;

            if (j > i && dist < r0 * 2.0 && lineIndex < maxConnections) {
                const li = lineIndex * 6;
                linePositions[li] = ix; linePositions[li+1] = iy; linePositions[li+2] = iz;
                linePositions[li+3] = x[j*3]; linePositions[li+4] = x[j*3+1]; linePositions[li+5] = x[j*3+2];
                
                lineColors[li] = 0; lineColors[li+1] = 2.0; lineColors[li+2] = 5.0; 
                lineColors[li+3] = 0; lineColors[li+4] = 2.0; lineColors[li+5] = 5.0;
                lineIndex++;
            }
          }
      }

      let particleDamping = 0.85; 
      
      if (!started) {
          particleDamping = 0.95; 
      } else if (effectiveChaos) {
          particleDamping = 0.98;
      } else if (isEncoding) {
          if (isTarget) {
              const dx = target[idx3] - x[idx3];
              const dy = target[idx3+1] - x[idx3+1];
              const dz = target[idx3+2] - x[idx3+2];
              const distSq = dx*dx + dy*dy + dz*dz;
              particleDamping = distSq < 0.1 ? 0.05 : 0.50;
          } else {
              particleDamping = 0.20; 
          }
      } else if (isRecalling) {
          particleDamping = 0.15;
      }

      v[idx3] = v[idx3] * particleDamping + fx;
      v[idx3+1] = v[idx3+1] * particleDamping + fy;
      v[idx3+2] = v[idx3+2] * particleDamping + fz;
      
      const speedSq = v[idx3]**2 + v[idx3+1]**2 + v[idx3+2]**2;
      const speed = Math.sqrt(speedSq);
      totalSpeed += speed;
      totalKineticEnergy += 0.5 * speedSq; // E_k = 0.5 * m * v^2 (m=1)

      if (speedSq < 0.0001 && !effectiveChaos && started) {
           const jitter = isRecalling ? 0.0005 : 0.002;
           v[idx3] += (Math.random() - 0.5) * jitter;
           v[idx3+1] += (Math.random() - 0.5) * jitter;
           v[idx3+2] += (Math.random() - 0.5) * jitter;
      }

      x[idx3] += v[idx3]; x[idx3+1] += v[idx3+1]; x[idx3+2] += v[idx3+2];
      
      const rSq = x[idx3]**2 + x[idx3+1]**2 + x[idx3+2]**2;
      if (rSq > 2500) { x[idx3]*=0.99; x[idx3+1]*=0.99; x[idx3+2]*=0.99; }
      phase[i] += phaseSyncRate * phaseDelta + 0.05;

      // Update Matrices
      TEMP_OBJ.position.set(x[idx3], x[idx3+1], x[idx3+2]);
      const s = hasTarget[i] ? 0.25 : 0.35; 
      TEMP_OBJ.scale.set(s, s, s);
      TEMP_OBJ.updateMatrix();
      meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
      if(outlineRef.current) outlineRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      const entropy = Math.min(1.0, speed * 0.5); 
      let r=0, g=0, b=0;
      if (rid === 0) { r=0.1; g=1.0; b=1.0; } // Cyan
      else if (rid === 1) { r=1.0; g=0.1; b=1.0; } // Magenta
      else { r=1.0; g=0.8; b=0.0; } // Gold

      const coreMix = entropy * 2.0; 
      TEMP_COLOR.setRGB(r * coreMix, g * coreMix, b * coreMix); 
      
      if (speed < 0.008 && !effectiveChaos && (isTarget || isRecalling) && started) {
          const flash = 1.0 - (speed / 0.008); 
          TEMP_COLOR.lerp(WHITE, flash * 0.8);
      }
      meshRef.current.setColorAt(i, TEMP_COLOR);

      const pulse = 1.0 + Math.sin(phase[i]) * 0.3;
      const glowIntensity = 2.5 + entropy * 4.0; 
      TEMP_EMISSIVE.setRGB(r * glowIntensity * pulse, g * glowIntensity * pulse, b * glowIntensity * pulse);
      
      if (speed < 0.008 && !effectiveChaos && (isTarget || isRecalling) && started) {
          const flash = 1.0 - (speed / 0.008);
          TEMP_EMISSIVE.lerp(WHITE, flash);
      }

      if(outlineRef.current) outlineRef.current.setColorAt(i, TEMP_EMISSIVE);
    }

    if (statsRef.current) {
        statsRef.current.meanError = activeTargetCount > 0 ? totalError / activeTargetCount : 0;
        statsRef.current.meanSpeed = totalSpeed / count;
        statsRef.current.energy = totalKineticEnergy; // Update Energy Metric
        statsRef.current.fps = 1 / delta;
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    if (outlineRef.current) {
        outlineRef.current.instanceMatrix.needsUpdate = true;
        if (outlineRef.current.instanceColor) outlineRef.current.instanceColor.needsUpdate = true;
    }

    linesRef.current.geometry.setDrawRange(0, lineIndex * 2);
    linesRef.current.geometry.attributes.position.needsUpdate = true;
    linesRef.current.geometry.attributes.color.needsUpdate = true;
  });

  return (
    <>
      <InstancedMesh ref={ghostRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[1, 8, 8]} />
        <MeshBasicMaterial color="#ffffff" transparent opacity={0.05} wireframe />
      </InstancedMesh>

      <InstancedMesh ref={outlineRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[0.42, 12, 12]} />
        <MeshBasicMaterial 
            color="#ffffff" 
            transparent 
            opacity={0.6}
            side={THREE.BackSide} 
            blending={THREE.AdditiveBlending} 
            depthWrite={false}
            toneMapped={false}
        />
      </InstancedMesh>

      <InstancedMesh ref={meshRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[0.22, 16, 16]} /> 
        <MeshStandardMaterial 
            color="#050505" 
            roughness={0.2} 
            metalness={0.9} 
            emissive="#000000"
            toneMapped={false}
        /> 
      </InstancedMesh>

      <LineSegments ref={linesRef} geometry={lineGeometry}>
        <LineBasicMaterial vertexColors={true} transparent opacity={0.4} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </LineSegments>
    </>
  );
};

const RegionGuides: React.FC<{ params: SimulationParams }> = ({ params }) => {
    // Only render if showRegions is true
    if (!params.showRegions) return null;

    return (
        <group>
            {/* Region A (Cyan) */}
            <Cylinder args={[18, 18, 10, 6]} position={[-20, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshBasicMaterial color="#00FFFF" wireframe transparent opacity={0.15} />
            </Cylinder>
            {/* Region B (Pink) */}
            <Cylinder args={[18, 18, 10, 6]} position={[20, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshBasicMaterial color="#FF00AA" wireframe transparent opacity={0.15} />
            </Cylinder>
            {/* Region Associative (Gold) */}
            <Cylinder args={[15, 15, 20, 8]} position={[0, 0, 0]}>
                <meshBasicMaterial color="#FFAA00" wireframe transparent opacity={0.15} />
            </Cylinder>
        </group>
    );
};

// --- Matrix HUD ---
// Visualizes the connectivity/adjacency matrix in real-time
const MatrixHUD: React.FC<{ 
    spatialRefs: React.MutableRefObject<{ neighborList: Int32Array; neighborCounts: Int32Array; }>; 
    particleCount: number;
}> = ({ spatialRefs, particleCount }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;
        
        const updateInterval = setInterval(() => {
            const size = canvasRef.current!.width;
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, size, size);

            // Region Guidelines (Background)
            // Cyan: 0-25% | Magenta: 25-50% | Gold: 50-100%
            const p0 = 0;
            const p1 = size * 0.25;
            const p2 = size * 0.5;
            const p3 = size;
            
            ctx.fillStyle = 'rgba(6, 182, 212, 0.1)'; // Cyan low alpha
            ctx.fillRect(p0, p0, p1, p1);

            ctx.fillStyle = 'rgba(236, 72, 153, 0.1)'; // Pink low alpha
            ctx.fillRect(p1, p1, p2-p1, p2-p1);

            ctx.fillStyle = 'rgba(234, 179, 8, 0.1)'; // Gold low alpha
            ctx.fillRect(p2, p2, p3-p2, p3-p2);

            // Draw connectivity
            const scale = size / particleCount;
            ctx.fillStyle = '#06b6d4'; // Cyan default

            // FIX: Sample across the entire population, not just the first 300
            // Stride to maintain performance while showing full structure
            const step = Math.max(1, Math.floor(particleCount / 300)); 

            for(let i=0; i<particleCount; i+=step) {
                const count = spatialRefs.current.neighborCounts[i];
                const offset = i * 48; // maxNeighbors
                const x = Math.floor(i * scale);
                
                // Draw self (diagonal)
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(x, x, 1, 1);

                // Set color based on Source Region
                if (i < particleCount * 0.25) ctx.fillStyle = '#06b6d4';
                else if (i < particleCount * 0.5) ctx.fillStyle = '#ec4899';
                else ctx.fillStyle = '#eab308';

                for(let n=0; n<count; n++) {
                    const j = spatialRefs.current.neighborList[offset + n];
                    if (j < particleCount) {
                        const y = Math.floor(j * scale);
                        ctx.fillRect(x, y, 1, 1);
                        ctx.fillRect(y, x, 1, 1); // Symmetric
                    }
                }
            }
        }, 100);

        return () => clearInterval(updateInterval);
    }, [particleCount]);

    return (
        <div className="absolute bottom-10 left-3 p-1 bg-black/80 border border-cyan-500/30 backdrop-blur-md">
            <div className="text-[9px] text-cyan-500 font-mono mb-1 tracking-widest">SYNAPTIC_DENSITY</div>
            <canvas ref={canvasRef} width={120} height={120} className="w-[120px] h-[120px] opacity-90" />
        </div>
    );
};

// --- Stress Report Modal ---
interface TestResult {
    pattern: string;
    finalError: number;
    convergenceSpeed: number;
}

const StressReportModal: React.FC<{ 
    results: TestResult[], 
    onClose: () => void 
}> = ({ results, onClose }) => {
    const avgError = results.reduce((acc, curr) => acc + curr.finalError, 0) / results.length;
    const capacityScore = Math.max(0, 100 - (avgError * 5)); // Arbitrary score calculation
    
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4">
            <div className="bg-black border-2 border-purple-500 w-full max-w-2xl p-8 relative shadow-[0_0_50px_rgba(168,85,247,0.4)] flex flex-col max-h-[90vh]">
                <h2 className="text-3xl font-black text-purple-400 mb-6 tracking-tighter border-b border-purple-900 pb-4">
                    MEMORY CAPACITY REPORT
                </h2>
                
                <div className="grid grid-cols-2 gap-8 mb-8 shrink-0">
                    <div>
                        <div className="text-xs text-purple-300 uppercase tracking-widest mb-1">Overall Capacity Score</div>
                        <div className="text-5xl font-bold text-white">{capacityScore.toFixed(1)}/100</div>
                    </div>
                    <div>
                        <div className="text-xs text-purple-300 uppercase tracking-widest mb-1">Patterns Stored</div>
                        <div className="text-5xl font-bold text-white">{results.length}</div>
                    </div>
                </div>

                <div className="mb-8 flex-1 overflow-hidden flex flex-col min-h-0 border border-purple-900/30">
                    <div className="bg-purple-900/20 px-4 py-2 border-b border-purple-800 shrink-0 grid grid-cols-3 text-sm text-purple-400 font-bold">
                        <div>PATTERN</div>
                        <div>RECALL ERROR</div>
                        <div>STATUS</div>
                    </div>
                    <div className="overflow-y-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-black p-4">
                        <table className="w-full text-sm text-left">
                             <tbody className="font-mono text-gray-300">
                                {results.map((r, i) => (
                                    <tr key={i} className="border-b border-purple-900/20 last:border-0 hover:bg-purple-900/10">
                                        <td className="py-2 w-1/3">{r.pattern}</td>
                                        <td className="py-2 w-1/3">{r.finalError.toFixed(2)}</td>
                                        <td className="py-2 w-1/3">
                                            <span className={r.finalError < 2.0 ? "text-green-400" : "text-red-400"}>
                                                {r.finalError < 2.0 ? "INTACT" : "DEGRADED"}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <button onClick={onClose} className="w-full py-4 bg-purple-900 hover:bg-purple-700 text-white font-bold tracking-widest border border-purple-500 transition-all shrink-0">
                    ACKNOWLEDGE REPORT
                </button>
            </div>
        </div>
    );
};


// --- UI Overlay ---

interface UIOverlayProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  dataRef: React.MutableRefObject<ParticleData>;
  simulationMode: 'standard' | 'stress';
  statsRef: React.MutableRefObject<SystemStats>;
  onTestComplete: (results: TestResult[]) => void;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams, dataRef, simulationMode, statsRef, onTestComplete }) => {
  const [showHelp, setShowHelp] = useState(false);
  const [autoRunPhase, setAutoRunPhase] = useState<'idle' | 'reset' | 'entropy' | 'observation' | 'encoding' | 'amnesia' | 'recall'>('idle');
  const [testLogs, setTestLogs] = useState<string[]>([]);

  const handleChange = (key: keyof SimulationParams, value: number | string | object | boolean) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const togglePause = () => handleChange('paused', !params.paused);
  
  const togglePlasticity = (active: boolean) => {
      if (active) setParams(prev => ({ 
          ...prev, 
          plasticity: 0.1, 
          dataGravity: Math.max(prev.dataGravity, 0.5) 
      }));
      else setParams(prev => ({ ...prev, plasticity: 0 }));
  };

  const handleMemoryAction = (type: 'save' | 'load', slot: number) => {
    setParams(prev => ({ ...prev, memoryAction: { type, slot, triggerId: prev.memoryAction.triggerId + 1 } }));
  };

  const addLog = (msg: string) => {
      setTestLogs(prev => [...prev.slice(-4), `> ${msg}`]);
  };

  // --- AUTOMATED STRESS TEST SEQUENCE (UPDATED 25-SLOT) ---
  useEffect(() => {
    if (simulationMode !== 'stress') return;

    let timeoutIds: ReturnType<typeof setTimeout>[] = [];
    const schedule = (fn: () => void, delay: number) => {
        timeoutIds.push(setTimeout(fn, delay));
    };

    // 25 Patterns (NATO Phonetic)
    const patterns = [
        "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", 
        "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIETT", 
        "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", 
        "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", 
        "UNIFORM", "VICTOR", "WHISKEY", "XRAY", "YANKEE"
    ];

    const results: TestResult[] = [];

    // T+0: Initialization
    setTestLogs(["> INIT STRESS PROTOCOL: 25 SLOTS (HIGH DENSITY)"]);
    handleMemoryAction('load', -2); // Clear bank
    setParams(p => ({ ...DEFAULT_PARAMS, chaosMode: false, showRegions: true, particleCount: p.particleCount }));

    let timeCursor = 1000;
    const learnTime = 3000; // Fast Learning
    const saveTime = 200;   // Instant Save
    
    // --- PHASE 1: ENCODING ---
    patterns.forEach((pat, index) => {
        // Learn
        schedule(() => {
            addLog(`ENCODING [${index + 1}/25]: '${pat}'`);
            setParams(p => ({ ...p, inputText: pat, plasticity: 0.2, dataGravity: 0.5 }));
        }, timeCursor);
        timeCursor += learnTime;

        // Save
        schedule(() => {
            handleMemoryAction('save', index + 1);
        }, timeCursor);
        timeCursor += saveTime;
    });

    // --- PHASE 2: RECALL & VERIFICATION ---
    schedule(() => {
        addLog("PHASE 2: VERIFICATION & RECALL");
        setParams(p => ({ ...p, inputText: "", plasticity: 0 })); // Stop external drive
    }, timeCursor);
    timeCursor += 1000;

    const recallTime = 2000; // Fast Recall

    patterns.forEach((pat, index) => {
        // Load
        schedule(() => {
            addLog(`RECALLING SLOT ${index + 1} ('${pat}')...`);
            handleMemoryAction('load', index + 1);
        }, timeCursor);
        timeCursor += recallTime;

        // Measure
        schedule(() => {
            const error = statsRef.current.meanError;
            results.push({
                pattern: pat,
                finalError: error,
                convergenceSpeed: statsRef.current.meanSpeed
            });
            // Only log errors to keep console clean
            if (error > 2.0) addLog(`WARNING: SLOT ${index+1} DEGRADED`);
        }, timeCursor);
        timeCursor += 200;
    });

    // --- PHASE 3: CROSSTALK ---
    schedule(() => {
        addLog("PHASE 3: TOTAL CROSSTALK SUPERPOSITION");
        handleMemoryAction('load', -1);
    }, timeCursor);
    timeCursor += 5000;

    // --- FINISH ---
    schedule(() => {
        addLog("TEST COMPLETE. GENERATING REPORT...");
        onTestComplete(results);
    }, timeCursor);


    return () => timeoutIds.forEach(clearTimeout);
  }, [simulationMode]);


  // Step-by-Step Experiment Controller (Standard Mode Only)
  useEffect(() => {
    if (simulationMode === 'stress') return; // Disable in stress mode
    if (autoRunPhase === 'idle') return;

    if (autoRunPhase === 'reset') {
        setParams(DEFAULT_PARAMS);
    } 
    else if (autoRunPhase === 'entropy') {
        setParams(prev => ({ ...prev, chaosMode: true, inputText: "" }));
    }
    else if (autoRunPhase === 'observation') {
        const word = "QUANTUM";
        setParams(prev => ({ ...prev, chaosMode: false, inputText: word, dataGravity: 0.4 }));
    }
    else if (autoRunPhase === 'encoding') {
        togglePlasticity(true); 
        setTimeout(() => handleMemoryAction('save', 1), 1000); 
    }
    else if (autoRunPhase === 'amnesia') {
        togglePlasticity(false); 
        setParams(prev => ({ ...prev, inputText: "", chaosMode: true }));
    }
    else if (autoRunPhase === 'recall') {
        setParams(prev => ({ ...prev, chaosMode: false }));
        handleMemoryAction('load', 1);
    }
  }, [autoRunPhase, simulationMode]);

  const nextStep = () => {
      const phases: typeof autoRunPhase[] = ['idle', 'reset', 'entropy', 'observation', 'encoding', 'amnesia', 'recall'];
      const currentIndex = phases.indexOf(autoRunPhase);
      const nextIndex = (currentIndex + 1) % phases.length;
      setAutoRunPhase(phases[nextIndex]);
  };

  const getNextButtonText = () => {
      switch(autoRunPhase) {
          case 'idle': return "START EXPERIMENT";
          case 'reset': return "INJECT CHAOS >>";
          case 'entropy': return "OBSERVE INPUT >>";
          case 'observation': return "ENCODE PATTERN >>";
          case 'encoding': return "INDUCE AMNESIA >>";
          case 'amnesia': return "ATTEMPT RECALL >>";
          case 'recall': return "FINISH >>";
          default: return "NEXT >>";
      }
  };

  const getStandardStatusText = () => {
    switch(autoRunPhase) {
        case 'idle': return "SYSTEM READY";
        case 'reset': return "SYSTEM_RESET: MEMORY WIPED";
        case 'entropy': return "PHASE 1: ENTROPY INJECTION";
        case 'observation': return 'PHASE 2: OBSERVATION (Input: "QUANTUM")';
        case 'encoding': return "PHASE 3: ENCODING (Annealing)";
        case 'amnesia': return "PHASE 4: AMNESIA (Scrambling)";
        case 'recall': return "PHASE 5: RECALL (Free Energy Min)";
        default: return "";
    }
  };

  const panelClass = "bg-black/80 backdrop-blur-md border border-cyan-500/30 p-3 text-cyan-100 shadow-[0_0_15px_rgba(6,182,212,0.15)] relative overflow-hidden";

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end font-['Rajdhani'] pb-8">
      {/* SCANLINES OVERLAY */}
      <div className="scanlines"></div>

      <div className={`${panelClass} pointer-events-auto flex flex-col gap-3 w-full clip-path-polygon`}>
        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-400"></div>
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-400"></div>

        <div className="flex justify-between items-center border-b border-cyan-900/50 pb-2">
            <h1 className="text-lg font-bold text-cyan-400 tracking-widest drop-shadow-[0_0_5px_rgba(34,211,238,0.8)]">NEURO_HOLOGRAPHIC</h1>
            <div className="flex items-center gap-2">
                 <button onClick={togglePause} className={`p-1.5 border border-cyan-500/50 ${params.paused ? 'text-yellow-400 animate-pulse' : 'text-cyan-600'}`}>
                    {params.paused ? "||" : ">>"}
                 </button>
                 <button onClick={() => setShowHelp(true)} className="text-cyan-600 hover:text-cyan-300">?</button>
            </div>
        </div>

        {/* CONTROLLER PANEL */}
        <div className="space-y-3 pt-2">
            <div className="bg-cyan-950/30 border border-cyan-500/20 p-3 text-center relative">
                {simulationMode === 'stress' ? (
                     <div className="text-left">
                        <div className="flex items-center gap-2 mb-2 border-b border-purple-800 pb-2">
                            <div className="w-2 h-2 bg-purple-500 animate-pulse"></div>
                            <span className="text-xs text-purple-300 font-bold tracking-widest">MODE: STRESS TEST (25-SLOT)</span>
                        </div>
                        
                        {/* Real-time Feedback Console */}
                        <div className="text-[10px] text-purple-200 font-mono bg-black/80 p-2 border border-purple-500/30 min-h-[80px] flex flex-col justify-end">
                            {testLogs.map((log, i) => (
                                <div key={i} className="opacity-80">{log}</div>
                            ))}
                            <div className="w-2 h-4 bg-purple-500 animate-pulse inline-block mt-1"></div>
                        </div>
                    </div>
                ) : (
                    /* STANDARD MODE CONTROLS */
                    autoRunPhase === 'idle' ? (
                        <button onClick={() => setAutoRunPhase('reset')} className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-sm tracking-widest shadow-[0_0_20px_rgba(6,182,212,0.6)] animate-pulse clip-corner transition-all">
                            INITIALIZE EXPERIMENT
                        </button>
                    ) : (
                        <div className="text-left">
                             <div className="flex items-center gap-2 mb-2 border-b border-cyan-800 pb-2">
                                <div className="w-2 h-2 bg-green-400 animate-pulse"></div>
                                <span className="text-xs text-cyan-300 font-bold tracking-widest">PHASE: {autoRunPhase.toUpperCase()}</span>
                             </div>
                             
                             <p className="text-xs text-white font-mono bg-black/50 p-2 border-l-2 border-cyan-500 mb-3 min-h-[40px]">
                                {getStandardStatusText()}
                             </p>
    
                             <button onClick={nextStep} className="w-full py-2 bg-cyan-700 hover:bg-cyan-500 text-white font-bold text-xs tracking-widest border border-cyan-400/50 transition-all flex justify-between px-4 items-center group">
                                <span>{getNextButtonText()}</span>
                                <span className="group-hover:translate-x-1 transition-transform">>></span>
                             </button>
                        </div>
                    )
                )}
            </div>
        </div>
        
        {/* Tools */}
        <div className="mt-2 pt-2 border-t border-cyan-900/50 space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-[10px] text-cyan-500 font-bold uppercase">Regions Viz</span>
                <button onClick={() => handleChange('showRegions', !params.showRegions)} className={`px-2 py-1 text-[9px] border ${params.showRegions ? 'bg-cyan-500 text-black border-cyan-400' : 'text-cyan-600 border-cyan-800'}`}>
                    {params.showRegions ? "VISIBLE" : "HIDDEN"}
                </button>
            </div>
        </div>

      </div>
      
      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 pointer-events-auto">
            <div className="bg-black border border-cyan-500 rounded-none w-full max-w-lg p-6 relative shadow-[0_0_30px_rgba(6,182,212,0.3)]">
                <button onClick={() => setShowHelp(false)} className="absolute top-2 right-2 text-cyan-600 hover:text-cyan-200">X</button>
                <h2 className="text-xl font-bold text-cyan-400 mb-4 tracking-widest border-b border-cyan-800 pb-2">OPERATOR_MANUAL</h2>
                <div className="space-y-2 text-xs text-cyan-100 font-mono">
                    <p className="text-cyan-300">>> MODE: CYCLIC DEMO</p>
                    <p>Automated sequence: Entropy Injection -> Pattern Formation -> Hebbian Learning -> Dissolution.</p>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

// --- Status Bar ---

const StatusBar: React.FC<{ params: SimulationParams, statsRef: React.MutableRefObject<SystemStats> }> = ({ params, statsRef }) => {
  const [stats, setStats] = useState({ meanError: 0, meanSpeed: 0, energy: 0, fps: 0 });

  useEffect(() => {
    const interval = setInterval(() => {
        if (statsRef.current) {
            setStats({
                meanError: statsRef.current.meanError,
                meanSpeed: statsRef.current.meanSpeed,
                energy: statsRef.current.energy,
                fps: statsRef.current.fps
            });
        }
    }, 200); 
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute bottom-0 left-0 w-full h-6 bg-cyan-950/90 border-t border-cyan-800 flex items-center px-3 text-[11px] font-mono text-cyan-400 gap-6 select-none z-50 backdrop-blur-sm">
        <div className="flex items-center gap-2">
            <span className="opacity-50">STATUS:</span>
            <span className={stats.meanError < 0.5 ? "text-green-400" : "text-yellow-400"}>
                {stats.meanError < 0.5 ? "STABLE" : "CONVERGING"}
            </span>
        </div>
        <div className="flex items-center gap-2">
            <span className="opacity-50">CONVERGENCE:</span>
            <span>{Math.round(Math.max(0, (1 - stats.meanSpeed * 5) * 100))}%</span>
        </div>
        <div className="flex items-center gap-2">
             <span className="opacity-50">ENERGY (Ek):</span>
             <span className={stats.energy > 5 ? "text-orange-400 animate-pulse" : "text-cyan-200"}>
                 {stats.energy.toFixed(2)}
             </span>
        </div>
        <div className="flex items-center gap-2">
            <span className="opacity-50">ENTROPY:</span>
            <span>{params.chaosMode ? 'MAX' : 'NOMINAL'}</span>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2">
            <span className="opacity-50">FPS:</span>
            <span>{Math.round(stats.fps)}</span>
        </div>
    </div>
  )
}

// --- Title Screen ---

const TitleScreen: React.FC<{ onStartStandard: () => void, onStartStress: () => void }> = ({ onStartStandard, onStartStress }) => {
    return (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
            <div className="max-w-3xl w-full p-12 border-l-4 border-cyan-500 bg-black/90 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-30 text-[10px] text-cyan-500 font-mono text-right">
                    REF: L-GROUP-PCN-2025<br/>
                    VAR_FREE_ENERGY_MIN
                </div>
                
                <h1 className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-white to-purple-500 mb-2 tracking-tighter" style={{ fontFamily: 'Rajdhani' }}>
                    PREDICTIVE MORPHOLOGY
                </h1>
                <h2 className="text-xl text-cyan-600 font-bold tracking-[0.5em] mb-8 uppercase">L-Group Simulation Environment</h2>
                
                <div className="mb-8 text-gray-300 font-mono text-sm leading-relaxed border-t border-b border-gray-800 py-6">
                    <strong className="text-white block mb-2">ABSTRACT</strong>
                    This experiment demonstrates the emergence of auto-associative memory through thermodynamic annealing. By minimizing the Variational Free Energy (F = Ep + Ek), the system transitions from entropic chaos to ordered semantic states. Phase 3 (Encoding) utilizes Hebbian-modulated structural plasticity to encode sensory data into the manifold, demonstrating proof of convergence to local minima as predicted by the L-Group framework.
                </div>
                
                <div className="flex gap-4">
                    <button onClick={onStartStandard} className="flex-1 py-4 bg-cyan-600 hover:bg-cyan-500 text-black font-bold text-lg tracking-widest clip-corner transition-all shadow-[0_0_30px_rgba(6,182,212,0.4)] hover:shadow-[0_0_50px_rgba(6,182,212,0.6)]">
                        INITIALIZE SIMULATION
                    </button>
                    
                    <button onClick={onStartStress} className="flex-1 py-4 bg-purple-900/50 hover:bg-purple-700/80 text-purple-200 border border-purple-500 font-bold text-lg tracking-widest clip-corner transition-all shadow-[0_0_20px_rgba(168,85,247,0.2)]">
                        STRESS TEST (CAPACITY)
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Main App ---

const SimulationCanvas: React.FC<{ 
    params: SimulationParams, 
    dataRef: React.MutableRefObject<ParticleData>, 
    statsRef: React.MutableRefObject<SystemStats>, 
    started: boolean,
    spatialRefs: React.MutableRefObject<{ neighborList: Int32Array; neighborCounts: Int32Array; gridHead: Int32Array; gridNext: Int32Array; frameCounter: number; }>
}> = ({ params, dataRef, statsRef, started, spatialRefs }) => {
  return (
    <Canvas camera={{ position: [0, 0, 35], fov: 45 }} gl={{ antialias: false, toneMapping: THREE.NoToneMapping }}>
        <color attach="background" args={['#020205']} />
        
        {/* Volumetric Atmosphere */}
        <fog attach="fog" args={['#020205', 10, 80]} />
        <ambientLight intensity={0.2} />
        
        <pointLight position={[20, 20, 20]} intensity={2} color="#00ffff" distance={100} decay={2} />
        <pointLight position={[-20, -10, -20]} intensity={2} color="#ff00ff" distance={100} decay={2} />
        
        <Stars radius={150} depth={50} count={3000} factor={4} saturation={1} fade speed={2} />
        <Grid position={[0, -15, 0]} args={[100, 100]} cellSize={4} sectionSize={20} sectionColor="#06b6d4" cellColor="#1e293b" fadeDistance={60} />

        <ParticleSystem params={params} dataRef={dataRef} statsRef={statsRef} started={started} spatialRefs={spatialRefs} />
        <RegionGuides params={params} />
        
        <EffectComposer enableNormalPass={false}>
            <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} radius={0.5} />
            <ChromaticAberration offset={[new THREE.Vector2(0.002, 0.002)] as any} radialModulation={false} modulationOffset={0} />
            <Noise opacity={0.05} />
            <Vignette eskil={false} offset={0.1} darkness={1.1} />
        </EffectComposer>

        <OrbitControls autoRotate={!started} autoRotateSpeed={0.5} maxPolarAngle={Math.PI / 1.5} minPolarAngle={Math.PI / 4} />
    </Canvas>
  );
};

function App() {
  // null = title screen, 'standard' = normal, 'stress' = automated stress test
  const [simulationMode, setSimulationMode] = useState<'standard' | 'stress' | null>(null);
  const [testResults, setTestResults] = useState<TestResult[] | null>(null);
  
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  const statsRef = useRef<SystemStats>({ meanError: 0, meanSpeed: 0, energy: 0, fps: 0 });
  
  // Hoist spatial refs to share with HUD
  const spatialRefs = useRef({
      gridHead: new Int32Array(4096),
      gridNext: new Int32Array(0),
      neighborList: new Int32Array(0),
      neighborCounts: new Int32Array(0),
      frameCounter: 0
  });

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

  const handleTestComplete = (results: TestResult[]) => {
      setTestResults(results);
  };

  const handleCloseReport = () => {
      setTestResults(null);
      setSimulationMode(null); // Return to title
  };

  return (
    <div className="w-full h-screen bg-black overflow-hidden relative font-sans select-none">
      <SimulationCanvas params={params} dataRef={dataRef} statsRef={statsRef} started={simulationMode !== null} spatialRefs={spatialRefs} />
      
      {simulationMode === null && (
          <TitleScreen 
            onStartStandard={() => setSimulationMode('standard')} 
            onStartStress={() => setSimulationMode('stress')} 
          />
      )}
      
      {simulationMode !== null && (
          <>
            <UIOverlay 
                params={params} 
                setParams={setParams} 
                dataRef={dataRef} 
                simulationMode={simulationMode} 
                statsRef={statsRef}
                onTestComplete={handleTestComplete}
            />
            <StatusBar params={params} statsRef={statsRef} />
            <MatrixHUD spatialRefs={spatialRefs} particleCount={params.particleCount} />
            <div className="absolute top-6 left-6 pointer-events-none hidden md:block mix-blend-screen">
                <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-white to-purple-500 tracking-tighter" style={{ fontFamily: 'Rajdhani' }}>
                PREDICTIVE_MORPHOLOGY
                </h1>
                <div className="flex items-center gap-2 mt-1">
                    <div className="w-2 h-2 bg-green-500 animate-pulse"></div>
                    <p className="text-xs text-cyan-600 font-mono">NET_V0.9.4 // ONLINE</p>
                </div>
            </div>
          </>
      )}

      {testResults && (
          <StressReportModal results={testResults} onClose={handleCloseReport} />
      )}
    </div>
  );
}

export default App;