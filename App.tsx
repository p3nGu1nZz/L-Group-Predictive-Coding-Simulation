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
  
  // Font Sizing - Kept reduced for composition
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

interface SpatialData {
  gridHead: Int32Array;
  gridNext: Int32Array;
  neighborList: Int32Array;
  neighborCounts: Int32Array;
  frameCounter: number;
}

interface ParticleSystemProps {
  params: SimulationParams;
  dataRef: React.MutableRefObject<ParticleData>;
  spatialRef: React.MutableRefObject<SpatialData>;
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

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef, spatialRef }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const outlineRef = useRef<THREE.InstancedMesh>(null); 
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  const memoryBank = useRef<Map<number, MemorySnapshot>>(new Map());
  
  const systemState = useRef({ meanError: 10.0 });
  const data = dataRef;

  // Initialization
  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    // Check if spatialRef needs initialization or resizing
    if (spatialRef.current.gridNext.length !== count) {
       spatialRef.current.gridNext = new Int32Array(count);
       spatialRef.current.neighborList = new Int32Array(count * 128); 
       spatialRef.current.neighborCounts = new Int32Array(count);
       spatialRef.current.gridHead = new Int32Array(4096);
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

  // Text Processing
  useEffect(() => {
    if (params.paused) return; 

    const count = params.particleCount;
    const { positions, count: pointCount } = textToPoints(params.inputText, Math.floor(count * 0.8));
    
    data.current.hasTarget.fill(0);

    if (pointCount > 0) {
        const targets = [];
        for(let i=0; i<pointCount; i++) {
            targets.push({ x: positions[i*3], y: positions[i*3+1], z: positions[i*3+2] });
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
  useFrame((state) => {
    if (!meshRef.current || !linesRef.current || !outlineRef.current || params.paused) return;

    const count = params.particleCount;

    // Fix: Lazily initialize instanceColor if it doesn't exist (it is null by default on InstancedMesh)
    if (!meshRef.current.instanceColor) {
        meshRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    }
    if (!outlineRef.current.instanceColor) {
        outlineRef.current.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    }

    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, plasticity, chaosMode } = params;
    const { x, v, phase, activation, target, hasTarget, memoryMatrix, regionID } = data.current;
    
    // Check data integrity
    if (x.length === 0) return;

    // OPTIMIZATION: Cache references to arrays to avoid object lookups in loop
    // These accessors are now safe because of the checks above
    const meshMatrices = meshRef.current.instanceMatrix.array;
    const meshColors = meshRef.current.instanceColor.array;
    const outlineMatrices = outlineRef.current.instanceMatrix.array;
    const outlineColors = outlineRef.current.instanceColor.array;
    const neighborList = spatialRef.current.neighborList;
    const neighborCounts = spatialRef.current.neighborCounts;

    let activeTargetCount = 0;
    for(let k = 0; k < count; k++) if(hasTarget[k] === 1) activeTargetCount++;
    const hasActiveTargets = activeTargetCount > 0;
    const damping = chaosMode ? 0.98 : (hasActiveTargets ? 0.6 : 0.90);

    // Spatial Hashing
    spatialRef.current.frameCounter++;
    const CELL_SIZE = 5.0;
    const GRID_SIZE = 4096; 
    
    // OPTIMIZATION: Refresh grid less often (every 4 frames instead of 3)
    if (spatialRef.current.frameCounter % 4 === 0) {
        const { gridHead, gridNext } = spatialRef.current;
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

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0;
      const idx3 = i * 3;
      const ix = x[idx3], iy = x[idx3 + 1], iz = x[idx3 + 2];
      const nOffset = i * 48; 
      const nCount = neighborCounts[i];

      if (chaosMode) {
          fx += (Math.random() - 0.5) * 1.5;
          fy += (Math.random() - 0.5) * 1.5;
          fz += (Math.random() - 0.5) * 1.5;
          fx += -iy * 0.05; fy += ix * 0.05;
      }

      if (hasTarget[i] && !chaosMode) {
          const dx = target[idx3] - ix;
          const dy = target[idx3+1] - iy;
          const dz = target[idx3+2] - iz;
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          totalError += dist;
          const k_spring = 0.2; 
          fx += dx * k_spring;
          fy += dy * k_spring;
          fz += dz * k_spring;
      }

      const interactionStrength = hasTarget[i] ? 0.1 : 1.0;
      if (interactionStrength > 0.01) {
          for (let n = 0; n < nCount; n++) {
            const j = neighborList[nOffset + n];
            const dx = x[j*3] - ix; const dy = x[j*3+1] - iy; const dz = x[j*3+2] - iz;
            const distSq = dx*dx + dy*dy + dz*dz;
            if (distSq < 0.01 || distSq > 16.0) continue; 
            const dist = Math.sqrt(distSq);
            const r0 = equilibriumDistance; 
            let force = 0;
            if (dist < r0) force = -stiffness * (r0 - dist) * 2.0;
            else if (!hasTarget[i]) force = stiffness * (dist - r0) * 0.1;
            
            force *= interactionStrength;
            const invDist = 1.0 / dist;
            fx += dx * invDist * force; fy += dy * invDist * force; fz += dz * invDist * force;
            const phaseDiff = phase[j] - phase[i];
            phaseDelta += Math.sin(phaseDiff) * 0.1;

            if (j > i && dist < r0 * 2.0 && lineIndex < maxConnections) {
                const li = lineIndex * 6;
                linePositions[li] = ix; linePositions[li+1] = iy; linePositions[li+2] = iz;
                linePositions[li+3] = x[j*3]; linePositions[li+4] = x[j*3+1]; linePositions[li+5] = x[j*3+2];
                
                // INTENSE LINE GLOW
                lineColors[li] = 0; lineColors[li+1] = 2.0; lineColors[li+2] = 5.0; 
                lineColors[li+3] = 0; lineColors[li+4] = 2.0; lineColors[li+5] = 5.0;
                lineIndex++;
            }
          }
      }

      v[idx3] = v[idx3] * damping + fx;
      v[idx3+1] = v[idx3+1] * damping + fy;
      v[idx3+2] = v[idx3+2] * damping + fz;
      x[idx3] += v[idx3]; x[idx3+1] += v[idx3+1]; x[idx3+2] += v[idx3+2];
      
      const rSq = x[idx3]**2 + x[idx3+1]**2 + x[idx3+2]**2;
      if (rSq > 2500) { x[idx3]*=0.99; x[idx3+1]*=0.99; x[idx3+2]*=0.99; }
      phase[i] += phaseSyncRate * phaseDelta + 0.05;

      // OPTIMIZATION: Direct Buffer Writing (Avoiding THREE.Object3D / THREE.Color overhead)
      // Matrix Layout: [sx, 0, 0, 0,  0, sy, 0, 0,  0, 0, sz, 0,  px, py, pz, 1]
      const s = hasTarget[i] ? 0.25 : 0.35; 
      const matIdx = i * 16;
      
      // Update Main Mesh Matrix
      meshMatrices[matIdx] = s;
      meshMatrices[matIdx+5] = s;
      meshMatrices[matIdx+10] = s;
      meshMatrices[matIdx+12] = x[idx3];
      meshMatrices[matIdx+13] = x[idx3+1];
      meshMatrices[matIdx+14] = x[idx3+2];
      meshMatrices[matIdx+15] = 1;

      // Update Outline Mesh Matrix
      outlineMatrices[matIdx] = s;
      outlineMatrices[matIdx+5] = s;
      outlineMatrices[matIdx+10] = s;
      outlineMatrices[matIdx+12] = x[idx3];
      outlineMatrices[matIdx+13] = x[idx3+1];
      outlineMatrices[matIdx+14] = x[idx3+2];
      outlineMatrices[matIdx+15] = 1;

      // --- COLOR DYNAMICS ---
      const speed = Math.sqrt(v[idx3]**2 + v[idx3+1]**2 + v[idx3+2]**2);
      const entropy = Math.min(1.0, speed * 0.5); 

      const rid = regionID[i];
      let r=0, g=0, b=0;
      if (rid === 0) { r=0.1; g=1.0; b=1.0; } 
      else if (rid === 1) { r=1.0; g=0.1; b=1.0; } 
      else { r=1.0; g=0.8; b=0.0; } 

      const coreMix = entropy * 2.0;
      const colIdx = i * 3;
      
      // Update Main Mesh Color
      meshColors[colIdx] = r * coreMix;
      meshColors[colIdx+1] = g * coreMix;
      meshColors[colIdx+2] = b * coreMix;

      // Update Outline Mesh Color
      const pulse = 1.0 + Math.sin(phase[i]) * 0.3;
      const glowIntensity = 2.5 + entropy * 4.0;
      outlineColors[colIdx] = r * glowIntensity * pulse;
      outlineColors[colIdx+1] = g * glowIntensity * pulse;
      outlineColors[colIdx+2] = b * glowIntensity * pulse;
    }

    systemState.current.meanError = activeTargetCount > 0 ? totalError / activeTargetCount : 0;
    
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

      {/* OUTLINE / GLOW */}
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

      {/* CORE MESH */}
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

interface SynapticMatrixProps {
  dataRef: React.MutableRefObject<ParticleData>;
  spatialRef: React.MutableRefObject<SpatialData>;
  count: number;
  params: SimulationParams;
}

const SynapticMatrix: React.FC<SynapticMatrixProps> = ({ dataRef, spatialRef, count, params }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef<Uint8ClampedArray | null>(null);
  const [expanded, setExpanded] = useState(false);
  const paramsRef = useRef(params);

  // Keep params current for the closure
  useEffect(() => { paramsRef.current = params; }, [params]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Initialize buffer if needed
    // 1200 * 1200 * 4 bytes = ~5.7 MB. Safe.
    const size = count * count * 4;
    if (!bufferRef.current || bufferRef.current.length !== size) {
        bufferRef.current = new Uint8ClampedArray(size);
        bufferRef.current.fill(0);
        // Set Alpha to 255 initially (opaque black)
        for (let i = 3; i < size; i+=4) bufferRef.current[i] = 255; 
        
        // Ensure exact 1:1 pixel match for matrix
        canvasRef.current.width = count;
        canvasRef.current.height = count;
        
        // Clear logic
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, count, count);
    }
    
    // Create ImageData object once we reuse
    const imgData = ctx.createImageData(count, count);

    // High performance visual loop (10 FPS)
    const draw = () => {
        if (!canvasRef.current || !bufferRef.current) return;
        
        // FREEZE LOGIC:
        // If in Chaos Mode (High Entropy), we STOP updating the buffer.
        // This preserves the pattern formed during the previous cycle (Persistence).
        if (paramsRef.current.chaosMode) {
             return; 
        }

        const { regionID } = dataRef.current;
        const { neighborList, neighborCounts } = spatialRef.current;
        const buffer = bufferRef.current;
        const width = count;

        // 1. FADE STEP (Bitwise Optimized)
        // Slower decay (0.96) allows trails to accumulate better.
        // Bitwise | 0 is faster than Math.floor
        for (let k = 0; k < buffer.length; k += 4) {
            const r = buffer[k], g = buffer[k+1], b = buffer[k+2];
            if (r > 0) buffer[k] = (r * 0.96) | 0;
            if (g > 0) buffer[k+1] = (g * 0.96) | 0;
            if (b > 0) buffer[k+2] = (b * 0.96) | 0;
            // Alpha stays 255
        }

        // 2. DRAW ACTIVE CONNECTIONS
        const maxNeighbors = 48; // Must match ParticleSystem constant
        
        for (let i = 0; i < count; i++) {
            const nCount = neighborCounts[i];
            const offset = i * maxNeighbors;
            const ri = regionID[i];

            for (let n = 0; n < nCount; n++) {
                const j = neighborList[offset + n];
                
                // Color Logic
                const rj = regionID[j];
                let r=255, g=255, b=255;
                if (ri === rj) {
                    if (ri === 0) { r=0; g=255; b=255; } 
                    else if (ri === 1) { r=255; g=0; b=255; } 
                    else { r=255; g=200; b=0; } 
                } else {
                     r=100; g=100; b=100;
                }

                // Plot (i, j)
                const idx1 = (i * width + j) * 4;
                if (r > buffer[idx1]) buffer[idx1] = r;
                if (g > buffer[idx1+1]) buffer[idx1+1] = g;
                if (b > buffer[idx1+2]) buffer[idx1+2] = b;

                // Plot (j, i) - Symmetric
                const idx2 = (j * width + i) * 4;
                if (r > buffer[idx2]) buffer[idx2] = r;
                if (g > buffer[idx2+1]) buffer[idx2+1] = g;
                if (b > buffer[idx2+2]) buffer[idx2+2] = b;
            }
        }

        // 3. UPLOAD TO CANVAS
        imgData.data.set(buffer);
        ctx.putImageData(imgData, 0, 0);
    };

    const interval = setInterval(draw, 100); // 10 FPS
    return () => clearInterval(interval);
  }, [count]);

  const download = (e: React.MouseEvent) => {
      e.stopPropagation();
      const link = document.createElement('a');
      link.download = 'synaptic_map.png';
      link.href = canvasRef.current!.toDataURL();
      link.click();
  };

  const wrapperClass = expanded 
    ? "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80vh] h-[80vh] z-[100] bg-black border-2 border-cyan-400 shadow-[0_0_100px_rgba(6,182,212,0.5)] p-2"
    : "absolute bottom-6 left-6 w-48 h-48 bg-black/90 border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.2)] cursor-pointer hover:border-cyan-400 hover:shadow-[0_0_25px_rgba(6,182,212,0.4)] transition-all overflow-hidden group";

  return (
    <>
      {expanded && <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[90]" onClick={() => setExpanded(false)} />}
      
      <div className={wrapperClass} onClick={() => !expanded && setExpanded(true)}>
         <canvas ref={canvasRef} className="w-full h-full object-contain bg-black" />
         
         {/* Overlays */}
         <div className="absolute top-0 left-0 w-full flex justify-between items-start p-2 pointer-events-none">
            <div className="bg-black/70 px-2 py-1 text-xs text-cyan-400 font-bold font-mono border-l-2 border-cyan-500">
                PERSISTENT_MEMORY_MAP
            </div>
            {expanded && (
                <div className="flex gap-2 pointer-events-auto">
                    <button onClick={download} className="bg-black/80 border border-cyan-500 text-cyan-400 px-3 py-1 text-xs hover:bg-cyan-500 hover:text-black font-bold transition-colors">
                        SAVE_IMG
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); setExpanded(false); }} className="bg-red-900/20 border border-red-500 text-red-500 px-3 py-1 text-xs hover:bg-red-500 hover:text-black font-bold transition-colors">
                        CLOSE
                    </button>
                </div>
            )}
         </div>
         
         {!expanded && (
            <div className="absolute bottom-0 right-0 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="text-[9px] text-cyan-300 bg-black/80 px-1 border border-cyan-500/30">EXPAND</div>
            </div>
         )}
      </div>
    </>
  );
};

// ... RegionGuides ... (Unchanged)
const RegionGuides: React.FC<{ params: SimulationParams }> = ({ params }) => {
    if (!params.showRegions) return null;
    return (
        <group>
            <Cylinder args={[14, 14, 7, 6]} position={[0, 10.5, 0]}>
                <meshBasicMaterial color="#00FFFF" wireframe transparent opacity={0.05} />
            </Cylinder>
            <Cylinder args={[14, 14, 7, 6]} position={[0, 3.5, 0]}>
                <meshBasicMaterial color="#FF00AA" wireframe transparent opacity={0.05} />
            </Cylinder>
            <Cylinder args={[14, 14, 14, 6]} position={[0, -7, 0]}>
                <meshBasicMaterial color="#FFAA00" wireframe transparent opacity={0.05} />
            </Cylinder>
        </group>
    );
};

// ... UIOverlay ... (Unchanged)
interface UIOverlayProps {
    params: SimulationParams;
    setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
    dataRef: React.MutableRefObject<ParticleData>;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams, dataRef }) => {
  const [localText, setLocalText] = useState("");
  const [showHelp, setShowHelp] = useState(false);
  const [activeTab, setActiveTab] = useState<'auto' | 'manual'>('auto');
  const [autoRunPhase, setAutoRunPhase] = useState<'idle' | 'reset' | 'chaos' | 'stabilize' | 'form' | 'learning' | 'save_dissolve' | 'loop_chaos'>('chaos');
  const [autoStatus, setAutoStatus] = useState("");

  const handleChange = (key: keyof SimulationParams, value: number | string | object | boolean) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const togglePause = () => handleChange('paused', !params.paused);
  const reset = () => { setParams(DEFAULT_PARAMS); setLocalText(""); };
  const handleTextSubmit = (e: React.FormEvent) => { e.preventDefault(); handleChange('inputText', localText); };
  const clearInput = () => { setLocalText(""); handleChange('inputText', ""); };
  const forgetCurrentMemory = () => handleChange('memoryResetTrigger', params.memoryResetTrigger + 1);

  const togglePlasticity = (active: boolean) => {
      if (active) setParams(prev => ({ ...prev, plasticity: 0.1, damping: 0.95, dataGravity: Math.max(prev.dataGravity, 0.5) }));
      else setParams(prev => ({ ...prev, plasticity: 0, damping: 0.85 }));
  };

  const handleMemoryAction = (type: 'save' | 'load', slot: number) => {
    setParams(prev => ({ ...prev, memoryAction: { type, slot, triggerId: prev.memoryAction.triggerId + 1 } }));
  };

  // Auto-Pilot Cycle - SPEED TUNED
  useEffect(() => {
    if (autoRunPhase === 'idle' || params.paused) return;
    let timer: ReturnType<typeof setTimeout>;

    if (autoRunPhase === 'reset') {
        setAutoStatus("SYSTEM_RESET");
        setParams(DEFAULT_PARAMS);
        setLocalText("");
        timer = setTimeout(() => setAutoRunPhase('chaos'), 500);
    } 
    else if (autoRunPhase === 'chaos') {
        setAutoStatus("ENTROPY_INJECTION_MODE");
        setParams(prev => ({ ...prev, chaosMode: true, inputText: "" }));
        timer = setTimeout(() => setAutoRunPhase('stabilize'), 4000); // 4s chaos (increased)
    }
    else if (autoRunPhase === 'stabilize') {
        setAutoStatus("FIELD_STABILIZATION");
        setParams(prev => ({ ...prev, chaosMode: false, inputText: "" }));
        timer = setTimeout(() => setAutoRunPhase('form'), 3000); // 3s stabilize (increased)
    }
    else if (autoRunPhase === 'form') {
        const word = "QUANTUM";
        setAutoStatus(`PATTERN_EMERGENCE: "${word}"`);
        setLocalText(word);
        setParams(prev => ({ ...prev, chaosMode: false, inputText: word, dataGravity: 0.4 }));
        timer = setTimeout(() => setAutoRunPhase('learning'), 8000); // 8s form/recall (significantly increased)
    }
    else if (autoRunPhase === 'learning') {
        setAutoStatus("NEURAL_CRYSTALLIZATION");
        togglePlasticity(true);
        timer = setTimeout(() => setAutoRunPhase('save_dissolve'), 6000); // 6s learning (doubled)
    }
    else if (autoRunPhase === 'save_dissolve') {
        setAutoStatus("STRUCTURE_DISSOLUTION");
        togglePlasticity(false);
        setLocalText("");
        setParams(prev => ({ ...prev, inputText: "" }));
        timer = setTimeout(() => setAutoRunPhase('chaos'), 2500); // 2.5s dissolve
    }
    return () => clearTimeout(timer);
  }, [autoRunPhase, params.paused]);

  const startAutoRun = () => setAutoRunPhase('reset');

  // Cyberpunk UI Classes
  const panelClass = "bg-black/80 backdrop-blur-md border border-cyan-500/30 p-3 text-cyan-100 shadow-[0_0_15px_rgba(6,182,212,0.15)] relative overflow-hidden";
  const btnClass = "bg-cyan-950/50 hover:bg-cyan-600/40 text-cyan-400 hover:text-cyan-100 border border-cyan-500/50 uppercase font-bold tracking-wider transition-all clip-corner";
  const labelClass = "text-[10px] text-cyan-500 font-bold uppercase tracking-widest mb-1";

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end font-['Rajdhani']">
      {/* SCANLINES OVERLAY */}
      <div className="scanlines"></div>

      <div className={`${panelClass} pointer-events-auto flex flex-col gap-3 w-full clip-path-polygon`}>
        {/* Decorative Corner */}
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

        {/* Tab Switcher */}
        <div className="flex border-b border-cyan-800">
            <button onClick={() => setActiveTab('auto')} className={`flex-1 py-1 text-xs font-bold uppercase ${activeTab === 'auto' ? 'bg-cyan-900/50 text-cyan-300 border-b-2 border-cyan-400' : 'text-cyan-800 hover:text-cyan-500'}`}>Auto_Cycle</button>
            <button onClick={() => setActiveTab('manual')} className={`flex-1 py-1 text-xs font-bold uppercase ${activeTab === 'manual' ? 'bg-purple-900/30 text-purple-300 border-b-2 border-purple-500' : 'text-cyan-800 hover:text-cyan-500'}`}>Manual_Override</button>
        </div>

        {/* AUTO VIEW */}
        {activeTab === 'auto' && (
            <div className="space-y-3 pt-2">
                <div className="bg-cyan-950/30 border border-cyan-500/20 p-3 text-center relative">
                    {autoRunPhase === 'idle' ? (
                        <button onClick={startAutoRun} className={`w-full py-2 ${btnClass} text-xs`}>INITIATE_SEQUENCE</button>
                    ) : (
                        <div className="text-left">
                             <div className="flex items-center gap-2 mb-1">
                                <div className="w-2 h-2 bg-cyan-400 animate-ping"></div>
                                <span className="text-xs text-cyan-300 font-bold tracking-widest">RUNNING_DAEMON</span>
                             </div>
                             <p className="text-xs text-white font-mono bg-black/50 p-1 border-l-2 border-cyan-500 pl-2">{autoStatus}</p>
                             {/* Progress Bar visual */}
                             <div className="h-1 w-full bg-black mt-2 overflow-hidden">
                                <div className="h-full bg-cyan-500 animate-progress"></div>
                             </div>
                        </div>
                    )}
                </div>
            </div>
        )}
        
        {/* MANUAL VIEW */}
        {activeTab === 'manual' && (
            <div className="space-y-3 pt-2">
                <div>
                  <div className="flex justify-between items-center mb-1">
                      <h2 className={labelClass}>DATA_INPUT</h2>
                      <button onClick={clearInput} className="text-[10px] text-red-400 hover:text-red-200 uppercase tracking-wider">[CLEAR_BUFFER]</button>
                  </div>
                  <form onSubmit={handleTextSubmit} className="flex gap-1 mb-2">
                    <input type="text" value={localText} onChange={(e) => setLocalText(e.target.value)} className="w-full bg-black border border-cyan-700 text-cyan-100 px-2 py-1 text-sm font-mono focus:border-cyan-400 outline-none" maxLength={10} />
                    <button type="submit" className={`${btnClass} px-3 text-xs`}>EXEC</button>
                  </form>
                  <div className="flex gap-1">
                     {[-1, 0, 1].map(r => (
                         <button key={r} onClick={() => handleChange('targetRegion', r)} className={`flex-1 text-[9px] py-1 border ${params.targetRegion === r ? 'bg-cyan-500 text-black border-cyan-400 font-bold' : 'border-cyan-900 text-cyan-700'}`}>
                             {r === -1 ? 'GLOBAL' : r === 0 ? 'SEC_A' : 'SEC_B'}
                         </button>
                     ))}
                  </div>
                </div>

                <div className="flex items-center justify-between border-t border-cyan-900/50 pt-2">
                    <div>
                        <div className="text-xs font-bold text-orange-400 uppercase tracking-widest">ENTROPY</div>
                    </div>
                    <button onClick={() => handleChange('chaosMode', !params.chaosMode)} className={`px-4 py-1 text-[10px] font-bold border ${params.chaosMode ? 'bg-orange-500 text-black border-orange-400 shadow-[0_0_10px_orange]' : 'bg-transparent text-gray-500 border-gray-800'}`}>
                        {params.chaosMode ? 'HIGH_ENERGY' : 'STABLE'}
                    </button>
                </div>

                <div className="flex items-center justify-between border-t border-cyan-900/50 pt-2">
                    <div>
                        <div className="text-xs font-bold text-yellow-400 uppercase tracking-widest">PLASTICITY</div>
                    </div>
                    <button onClick={() => togglePlasticity(params.plasticity === 0)} className={`px-4 py-1 text-[10px] font-bold border ${params.plasticity > 0 ? 'bg-yellow-500 text-black border-yellow-400 shadow-[0_0_10px_yellow]' : 'bg-transparent text-gray-500 border-gray-800'}`}>
                        {params.plasticity > 0 ? 'ACTIVE' : 'LOCKED'}
                    </button>
                </div>

                <div className="pt-2 border-t border-cyan-900/50">
                     <div className="flex justify-between items-center mb-1">
                        <h2 className="text-[10px] text-purple-400 font-bold uppercase tracking-widest">HOLO_MEMORY</h2>
                        <button onClick={forgetCurrentMemory} className="text-[9px] text-red-500 hover:text-white uppercase">[WIPE]</button>
                     </div>
                     <div className="grid grid-cols-4 gap-1">
                        {[1, 2, 3, 4].map(slot => (
                            <div key={slot} className="flex flex-col">
                                <button onClick={() => handleMemoryAction('load', slot)} className="bg-purple-900/40 hover:bg-purple-600 text-purple-200 border border-purple-500/50 text-[10px] py-1">R{slot}</button>
                                <button onClick={() => handleMemoryAction('save', slot)} className="bg-black text-purple-600 border border-purple-900 text-[8px] py-0.5 hover:text-white">SV</button>
                            </div>
                        ))}
                     </div>
                </div>
            </div>
        )}

        <button onClick={reset} className="w-full py-1 mt-2 bg-red-950/30 hover:bg-red-900/50 text-red-500 border border-red-800/50 text-[10px] uppercase tracking-[0.2em] transition-colors">
          SYSTEM_PURGE
        </button>
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
                    <p className="text-cyan-300 mt-4">>> MODE: MANUAL OVERRIDE</p>
                    <ul className="list-square pl-4 space-y-1 text-gray-400">
                        <li>Enter text string to seed attractor points.</li>
                        <li><span className="text-orange-400">HIGH_ENERGY:</span> Break static bonds, increase temperature.</li>
                        <li><span className="text-yellow-400">PLASTICITY:</span> Encode current spatial relations into memory matrix.</li>
                        <li><span className="text-purple-400">MEMORY R1-R4:</span> Recall saved states (associative hallucination).</li>
                    </ul>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

// --- Main App ---

const SimulationCanvas: React.FC<{ params: SimulationParams, dataRef: React.MutableRefObject<ParticleData>, spatialRef: React.MutableRefObject<SpatialData> }> = ({ params, dataRef, spatialRef }) => {
  return (
    <Canvas camera={{ position: [0, 0, 35], fov: 45 }} gl={{ antialias: false, toneMapping: THREE.NoToneMapping }}>
        <color attach="background" args={['#020205']} />
        
        {/* Volumetric Atmosphere */}
        <fog attach="fog" args={['#020205', 10, 80]} />
        <ambientLight intensity={0.2} />
        
        {/* Cyberpunk Lighting */}
        <pointLight position={[20, 20, 20]} intensity={2} color="#00ffff" distance={100} decay={2} />
        <pointLight position={[-20, -10, -20]} intensity={2} color="#ff00ff" distance={100} decay={2} />
        
        <Stars radius={150} depth={50} count={3000} factor={4} saturation={1} fade speed={2} />
        
        {/* Digital Floor Grid */}
        <Grid position={[0, -15, 0]} args={[100, 100]} cellSize={4} sectionSize={20} sectionColor="#06b6d4" cellColor="#1e293b" fadeDistance={60} />

        <ParticleSystem params={params} dataRef={dataRef} spatialRef={spatialRef} />
        <RegionGuides params={params} />
        
        {/* Post Processing for the GLOW */}
        <EffectComposer enableNormalPass={false}>
            <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} radius={0.5} />
            <ChromaticAberration offset={[new THREE.Vector2(0.002, 0.002)] as any} radialModulation={false} modulationOffset={0} />
            <Noise opacity={0.05} />
            <Vignette eskil={false} offset={0.1} darkness={1.1} />
        </EffectComposer>

        <OrbitControls autoRotate={false} maxPolarAngle={Math.PI / 1.5} minPolarAngle={Math.PI / 4} />
    </Canvas>
  );
};

function App() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
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

  const spatialRef = useRef<SpatialData>({
      gridHead: new Int32Array(0),
      gridNext: new Int32Array(0),
      neighborList: new Int32Array(0),
      neighborCounts: new Int32Array(0),
      frameCounter: 0
  });

  return (
    <div className="w-full h-screen bg-black overflow-hidden relative font-sans select-none">
      <SimulationCanvas params={params} dataRef={dataRef} spatialRef={spatialRef} />
      
      <UIOverlay params={params} setParams={setParams} dataRef={dataRef} />
      
      <SynapticMatrix dataRef={dataRef} spatialRef={spatialRef} count={params.particleCount} params={params} />

      {/* Futuristic Header */}
      <div className="absolute top-6 left-6 pointer-events-none hidden md:block mix-blend-screen">
        <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-white to-purple-500 tracking-tighter" style={{ fontFamily: 'Rajdhani' }}>
          PREDICTIVE_MORPHOLOGY
        </h1>
        <div className="flex items-center gap-2 mt-1">
             <div className="w-2 h-2 bg-green-500 animate-pulse"></div>
             <p className="text-xs text-cyan-600 font-mono">NET_V0.9.3 // ONLINE</p>
        </div>
      </div>
    </div>
  );
}

export default App;