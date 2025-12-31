import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SimulationParams, ParticleData } from '../types';

// Workaround for missing JSX types in current environment
const InstancedMesh = 'instancedMesh' as any;
const SphereGeometry = 'sphereGeometry' as any;
const MeshStandardMaterial = 'meshStandardMaterial' as any;
const MeshBasicMaterial = 'meshBasicMaterial' as any;
const LineSegments = 'lineSegments' as any;
const LineBasicMaterial = 'lineBasicMaterial' as any;

interface ParticleSystemProps {
  params: SimulationParams;
}

const TEMP_OBJ = new THREE.Object3D();
const TEMP_COLOR = new THREE.Color();

const textToPoints = (text: string): { positions: Float32Array, count: number } => {
  if (!text) return { positions: new Float32Array(0), count: 0 };

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { positions: new Float32Array(0), count: 0 };

  const fontSize = 60; 
  const width = 1024; 
  const height = 200;
  canvas.width = width;
  canvas.height = height;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = 'white';
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, height / 2);

  const imgData = ctx.getImageData(0, 0, width, height);
  const points: number[] = [];
  const step = 2; 

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const index = (y * width + x) * 4;
      if (imgData.data[index] > 128) {
        const px = (x - width / 2) * 0.12;
        const py = -(y - height / 2) * 0.12;
        const pz = 0; 
        points.push(px, py, pz);
      }
    }
  }
  
  return { positions: new Float32Array(points), count: points.length / 3 };
};

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);

  // Persistent Memory Bank: Stores full memory matrices (r0 values) for recall
  const memoryBank = useRef<Map<number, Float32Array>>(new Map());

  // Data refs
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

  // 1. Initialize Particles
  useEffect(() => {
    const count = params.particleCount;
    const memorySize = count * count;
    
    // Reset data structure when particle count changes
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

    // Initialize using Fibonacci Sphere algorithm for even distribution
    // This avoids the "pole clumping" seen in standard spherical coordinates
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle

    for (let i = 0; i < count; i++) {
      // y goes from 1 to -1
      const y = 1 - (i / (count - 1)) * 2; 
      // radius at this y
      const radiusAtY = Math.sqrt(1 - y * y); 
      const theta = phi * i;

      const x = Math.cos(theta) * radiusAtY;
      const z = Math.sin(theta) * radiusAtY;

      // Distribute radially (shell vs volume)
      // Cube root gives uniform volume distribution, but let's keep it somewhat shell-like for better visibility
      const r = 8 * Math.cbrt(Math.random() * 0.5 + 0.5); 

      data.current.x[i * 3] = x * r;
      data.current.x[i * 3 + 1] = y * r;
      data.current.x[i * 3 + 2] = z * r;

      data.current.phase[i] = Math.random() * Math.PI * 2;
      data.current.spin[i] = Math.random() > 0.5 ? 0.5 : -0.5;
    }
    
    // Clear bank on hard reset
    memoryBank.current.clear();

  }, [params.particleCount]);

  // 2. Handle Memory Bank Actions (Save/Load)
  useEffect(() => {
    const { type, slot } = params.memoryAction;
    if (type === 'idle') return;

    if (type === 'save') {
        // Deep copy the current memory matrix to the bank
        const currentMatrix = data.current.memoryMatrix;
        if (currentMatrix) {
            memoryBank.current.set(slot, new Float32Array(currentMatrix));
            console.log(`Saved state to Slot ${slot}`);
        }
    } else if (type === 'load') {
        // Retrieve and overwrite current matrix
        const savedMatrix = memoryBank.current.get(slot);
        if (savedMatrix && data.current.memoryMatrix) {
            data.current.memoryMatrix.set(savedMatrix);
            console.log(`Loaded state from Slot ${slot}`);
        }
    }
  }, [params.memoryAction]);

  // 3. Handle Memory Wipe
  useEffect(() => {
    if (params.memoryResetTrigger > 0) {
      if (data.current.memoryMatrix) {
        data.current.memoryMatrix.fill(-1);
      }
    }
  }, [params.memoryResetTrigger]);

  // 4. Handle Sensory Input (Global Optimal Assignment)
  useEffect(() => {
    const count = params.particleCount;
    const { positions, count: pointCount } = textToPoints(params.inputText);
    
    data.current.hasTarget.fill(0);

    if (pointCount > 0) {
        const currentPositions = data.current.x;
        
        // A. Prepare Targets
        const targets: {x: number, y: number, z: number}[] = [];
        if (pointCount > count) {
            // Downsample with stride to preserve shape
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
            // Use all target points
            for (let i = 0; i < pointCount; i++) {
                targets.push({
                    x: positions[i * 3],
                    y: positions[i * 3 + 1],
                    z: positions[i * 3 + 2]
                });
            }
        }

        // B. Global Minimum Distance Assignment
        const edges: { pIdx: number, tIdx: number, cost: number }[] = [];
        
        for (let i = 0; i < count; i++) {
           const px = currentPositions[i * 3];
           const py = currentPositions[i * 3 + 1];
           const pz = currentPositions[i * 3 + 2];
           
           for(let j = 0; j < targets.length; j++) {
              const tx = targets[j].x;
              const ty = targets[j].y;
              const tz = targets[j].z;
              const distSq = (px - tx)**2 + (py - ty)**2 + (pz - tz)**2;
              
              // Push edge
              edges.push({ pIdx: i, tIdx: j, cost: distSq });
           }
        }
        
        // Sort all potential connections by distance (closest first)
        edges.sort((a,b) => a.cost - b.cost);
        
        const assignedParticles = new Set<number>();
        const assignedTargets = new Set<number>();
        
        // Fill assignments based on shortest available links
        for (const edge of edges) {
           // If we have assigned all targets, stop
           if (assignedTargets.size === targets.length) break;
           
           // If neither particle nor target is taken, assign them
           if (!assignedParticles.has(edge.pIdx) && !assignedTargets.has(edge.tIdx)) {
              assignedParticles.add(edge.pIdx);
              assignedTargets.add(edge.tIdx);
              
              const t = targets[edge.tIdx];
              data.current.target[edge.pIdx * 3] = t.x;
              data.current.target[edge.pIdx * 3 + 1] = t.y;
              data.current.target[edge.pIdx * 3 + 2] = t.z;
              data.current.hasTarget[edge.pIdx] = 1;
              data.current.activation[edge.pIdx] = 1.0;
           }
        }
    }

    // Visual guide update (Ghost System)
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

  // --- PHYSICS LOOP ---
  useFrame((state) => {
    if (!meshRef.current || !linesRef.current) return;

    const { equilibriumDistance, stiffness, couplingDecay, phaseSyncRate, spatialLearningRate, dataGravity, plasticity, damping } = params;
    const count = params.particleCount;
    const { x, v, phase, activation, target, hasTarget, memoryMatrix } = data.current;
    
    // Adaptive Physics Logic
    // When plasticity is ON (learning mode), we:
    // 1. Lower the stiffness significantly so particles aren't held back by previous bonds
    // 2. Increase effective gravity so they snap to the text
    const isLearning = plasticity > 0;
    const adaptiveStiffness = isLearning ? stiffness * 0.2 : stiffness;
    const adaptiveGravity = isLearning ? Math.max(dataGravity, 0.6) : dataGravity;

    let lineIndex = 0;
    const linePositions = linesRef.current.geometry.attributes.position.array as Float32Array;
    const lineColors = linesRef.current.geometry.attributes.color.array as Float32Array;

    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0, fz = 0;
      let phaseDelta = 0;
      let stress = 0;

      const ix = x[i * 3];
      const iy = x[i * 3 + 1];
      const iz = x[i * 3 + 2];

      // 1. Particle-Particle Interactions
      for (let j = 0; j < count; j++) {
        if (i === j) continue;

        const jx = x[j * 3];
        const jy = x[j * 3 + 1];
        const jz = x[j * 3 + 2];

        const dx = jx - ix;
        const dy = jy - iy;
        const dz = jz - iz;
        const distSq = dx * dx + dy * dy + dz * dz;

        if (distSq > couplingDecay * couplingDecay * 1.5) continue;
        
        const dist = Math.sqrt(distSq);
        if (dist < 0.001) continue;

        const phaseDiff = phase[j] - phase[i];
        const couplingStrength = Math.exp(-distSq / (couplingDecay * couplingDecay));
        const vibrationalModulation = Math.cos(phaseDiff);
        
        // Memory Lookup
        const memIndex = i * count + j;
        let r0 = memoryMatrix[memIndex];
        
        // Plasticity: Learn current distance
        // Only learn if connected strongly enough
        if (isLearning && couplingStrength > 0.05) {
            // If unset, init to current distance or default
            if (r0 === -1) r0 = dist; 
            // Smoothly interpolate memory towards current distance
            r0 = r0 + (dist - r0) * plasticity;
            memoryMatrix[memIndex] = r0;
        }

        // Use global param if no memory exists
        if (r0 === -1) r0 = equilibriumDistance;

        // Force: Spring + Vibrational Modulation
        // weight is higher if particles are phase-synced
        const weight = couplingStrength * (0.6 + 0.4 * vibrationalModulation); 
        const forceMag = adaptiveStiffness * (dist - r0) * weight;
        
        stress += Math.abs(forceMag);

        const nx = dx / dist;
        const ny = dy / dist;
        const nz = dz / dist;

        fx += nx * forceMag;
        fy += ny * forceMag;
        fz += nz * forceMag;

        // Phase Sync
        if (couplingStrength > 0.1) phaseDelta += couplingStrength * Math.sin(phaseDiff);

        // Visuals
        if (j > i && couplingStrength > 0.1 && lineIndex < maxConnections) {
          const isLearned = memoryMatrix[memIndex] !== -1;
          
          linePositions[lineIndex * 6] = ix;
          linePositions[lineIndex * 6 + 1] = iy;
          linePositions[lineIndex * 6 + 2] = iz;

          linePositions[lineIndex * 6 + 3] = jx;
          linePositions[lineIndex * 6 + 4] = jy;
          linePositions[lineIndex * 6 + 5] = jz;

          let r=0.1, g=0.5, b=0.7; 
          if (isLearned) {
             r=1.0; g=0.84; b=0.0; // Gold
          }

          // Fade lines based on coupling
          const alpha = couplingStrength;

          lineColors[lineIndex * 6] = r * alpha;
          lineColors[lineIndex * 6 + 1] = g * alpha;
          lineColors[lineIndex * 6 + 2] = b * alpha;
          
          lineColors[lineIndex * 6 + 3] = r * alpha;
          lineColors[lineIndex * 6 + 4] = g * alpha;
          lineColors[lineIndex * 6 + 5] = b * alpha;

          lineIndex++;
        }
      }

      // 2. Sensory Input (Data Gravity)
      if (hasTarget[i] && params.inputText.length > 0) {
        const tx = target[i * 3];
        const ty = target[i * 3 + 1];
        const tz = target[i * 3 + 2];

        const dx = tx - ix;
        const dy = ty - iy;
        const dz = tz - iz;
        
        fx += dx * adaptiveGravity;
        fy += dy * adaptiveGravity;
        fz += dz * adaptiveGravity;
      }

      // 3. Integration
      activation[i] = activation[i] * 0.95 + stress * 0.05;

      // Apply Damping from params
      v[i * 3] = v[i * 3] * damping + fx * spatialLearningRate;
      v[i * 3 + 1] = v[i * 3 + 1] * damping + fy * spatialLearningRate;
      v[i * 3 + 2] = v[i * 3 + 2] * damping + fz * spatialLearningRate;

      x[i * 3] += v[i * 3];
      x[i * 3 + 1] += v[i * 3 + 1];
      x[i * 3 + 2] += v[i * 3 + 2];

      // Soft Container
      const rSq = x[i * 3]**2 + x[i * 3 + 1]**2 + x[i * 3 + 2]**2;
      if (rSq > 900) {
           const scale = 0.98;
           x[i * 3] *= scale;
           x[i * 3 + 1] *= scale;
           x[i * 3 + 2] *= scale;
      }

      phase[i] += phaseSyncRate * phaseDelta + 0.02;

      // Update Mesh
      TEMP_OBJ.position.set(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
      
      const energyLevel = Math.min(1.0, activation[i]); 
      const size = 0.4 + 0.4 * energyLevel;
      
      TEMP_OBJ.scale.set(size, size, size);
      TEMP_OBJ.updateMatrix();
      meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);

      // Color: Blue (calm) -> White (active) -> Gold (memory held)
      // We can use phase to modulate color slightly for "shimmer"
      const phaseColor = Math.sin(phase[i]) * 0.1;
      const r = 0.1 + energyLevel * 0.8;
      const g = 0.3 + energyLevel * 0.5 + phaseColor;
      const b = 0.8 + phaseColor;
      
      TEMP_COLOR.setRGB(r, g, b);
      meshRef.current.setColorAt(i, TEMP_COLOR);
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

export default ParticleSystem;