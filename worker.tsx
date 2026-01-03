
import * as THREE from 'three';
import { SimulationParams, ParticleData, MemorySnapshot, CONSTANTS, SystemStats } from './types';

// --- Constants & Colors ---
const SPIN_UP_COLOR = new THREE.Color("#ff0055");
const SPIN_DOWN_COLOR = new THREE.Color("#0055ff");
const WHITE = new THREE.Color(1, 1, 1);
const TEMP_COLOR = new THREE.Color();

// --- State ---
let params: SimulationParams | null = null;
let teacherFeedback = 0;
let started = false;

const data: ParticleData = {
  x: new Float32Array(0),
  v: new Float32Array(0),
  phase: new Float32Array(0),
  spin: new Int8Array(0),
  activation: new Float32Array(0),
  target: new Float32Array(0),
  hasTarget: new Uint8Array(0),
  memoryMatrix: new Float32Array(0),
  regionID: new Uint8Array(0),
  forwardMatrix: new Float32Array(0),
  feedbackMatrix: new Float32Array(0),
  delayedActivation: new Float32Array(0),
  lastActiveTime: new Float32Array(0),
  hysteresisState: new Uint8Array(0),
};

const spatial = {
    gridHead: new Int32Array(0),
    gridNext: new Int32Array(0),
    neighborList: new Int32Array(0),
    neighborCounts: new Int32Array(0),
    frameCounter: 0
};

const memoryBank = new Map<number, MemorySnapshot>();
const flashBuffer: Float32Array = new Float32Array(0);

// --- Math Helpers ---
const DyT = (x: number, alpha2: number, alpha3: number) => {
    return alpha2 * Math.tanh(alpha3 * x);
};

// --- Initialization ---
export function initSystem(count: number, initialParams: SimulationParams, feedback: number, isStarted: boolean) {
    const memorySize = count * count;
    
    params = initialParams;
    teacherFeedback = feedback;
    started = isStarted;

    // Resize Spatial Arrays
    if (spatial.gridHead.length === 0) {
         spatial.gridHead = new Int32Array(4096);
    }
    if (spatial.gridNext.length !== count) {
        spatial.gridNext = new Int32Array(count);
        spatial.neighborList = new Int32Array(count * 128);
        spatial.neighborCounts = new Int32Array(count);
    }

    // Resize Data Arrays
    if (data.x.length !== count * 3) {
        data.x = new Float32Array(count * 3);
        data.v = new Float32Array(count * 3);
        data.phase = new Float32Array(count);
        data.spin = new Int8Array(count);
        data.activation = new Float32Array(count);
        data.target = new Float32Array(count * 3);
        data.hasTarget = new Uint8Array(count);
        data.memoryMatrix = new Float32Array(memorySize).fill(-1);
        data.regionID = new Uint8Array(count);
        data.forwardMatrix = new Float32Array(memorySize);
        data.feedbackMatrix = new Float32Array(memorySize);
        data.delayedActivation = new Float32Array(count);
        data.lastActiveTime = new Float32Array(count);
        data.hysteresisState = new Uint8Array(count);
    }

    // Seed Particles
    const phi = Math.PI * (3 - Math.sqrt(5)); 
    for (let i = 0; i < count; i++) {
        const y = 1 - (i + 0.5) * (2 / count); 
        const radiusAtY = Math.sqrt(1 - y * y); 
        const theta = phi * i; 
        const r = 25.0 * Math.cbrt(Math.random()); 
        
        data.x[i * 3] = Math.cos(theta) * radiusAtY * r;
        data.x[i * 3 + 1] = y * r;
        data.x[i * 3 + 2] = Math.sin(theta) * radiusAtY * r;

        data.v[i * 3] = (Math.random() - 0.5) * 1.0;
        data.v[i * 3 + 1] = (Math.random() - 0.5) * 1.0;
        data.v[i * 3 + 2] = (Math.random() - 0.5) * 1.0;

        data.phase[i] = Math.random() * Math.PI * 2;
        data.spin[i] = Math.random() > 0.5 ? 1 : -1;

        if (i < Math.floor(count * 0.25)) data.regionID[i] = 0;      
        else if (i < Math.floor(count * 0.5)) data.regionID[i] = 1;  
        else data.regionID[i] = 2;                                   
    }
    
    // Do NOT clear memoryBank here to allow persistence across mode switches if needed,
    // or clear it if that's the desired behavior.
    // memoryBank.clear(); 
}

export function updateParams(newParams: SimulationParams, newStarted: boolean, newFeedback: number) {
    const oldTrigger = params?.memoryAction?.triggerId;
    const newTrigger = newParams.memoryAction?.triggerId;
    
    params = newParams;
    started = newStarted;
    teacherFeedback = newFeedback;
    
    // Handle Memory Action
    if (params && params.memoryAction && oldTrigger !== newTrigger) {
            const { type, slot } = params.memoryAction;
            if (type === 'save') {
                const snapshot: MemorySnapshot = {
                    x: new Float32Array(data.x),
                    regionID: new Uint8Array(data.regionID),
                    forwardMatrix: new Float32Array(data.forwardMatrix) 
                };
                memoryBank.set(slot, snapshot);
            } else if (type === 'load') {
                if (slot === -2) {
                    memoryBank.clear();
                    data.forwardMatrix.fill(0);
                } else {
                    const snap = memoryBank.get(slot);
                    if (snap) {
                        // Safe copy for target positions
                        const len = Math.min(data.target.length, snap.x.length);
                        if (len === data.target.length) {
                             data.target.set(snap.x);
                        } else {
                             // Partial copy or copy what fits
                             data.target.set(snap.x.subarray(0, len));
                        }
                        
                        data.hasTarget.fill(1);
                        data.v.fill(0);
                        
                        // Safe copy for forward matrix
                        if (snap.forwardMatrix) {
                            const matLen = Math.min(data.forwardMatrix.length, snap.forwardMatrix.length);
                            if (matLen === data.forwardMatrix.length) {
                                data.forwardMatrix.set(snap.forwardMatrix);
                            } else {
                                data.forwardMatrix.set(snap.forwardMatrix.subarray(0, matLen));
                            }
                        }
                    }
                }
            }
    }
}

export function updateTargets(targets: {x:number, y:number, z:number}[], indices: number[]) {
    if (!params) return;
    const targetCount = targets.length;
    const count = params.particleCount;
    
    data.hasTarget.fill(0);
    const assignCount = Math.min(count, targetCount);
    
    for (let i = 0; i < assignCount; i++) {
            const pid = indices[i];
            const t = targets[i];
            
            if (params.targetRegion !== -1 && data.regionID[pid] !== params.targetRegion) continue;
            
            // Boundary check
            if (pid * 3 + 2 < data.target.length) {
                data.target[pid * 3] = t.x;
                data.target[pid * 3 + 1] = t.y;
                data.target[pid * 3 + 2] = t.z;
                data.hasTarget[pid] = 1;
                data.activation[pid] = 1.0;
            }
    }
}

export function getMatrixData() {
    return data.forwardMatrix;
}

// --- Physics Step ---
export function step(dt: number) {
    if (!params) return null;

    const count = params.particleCount;
    const timeNow = performance.now() / 1000;
    const effectiveChaos = started ? params.chaosMode : false;
    
    // Teacher / Thermostat
    let systemTemperature = 0.0;
    if (teacherFeedback === 1) systemTemperature = 0.0; 
    else if (teacherFeedback === -1) systemTemperature = 1.0; 
    else systemTemperature = 0.05;

    const { equilibriumDistance, stiffness, plasticity, phaseSyncRate, usePaperPhysics, spinCouplingStrength, phaseCouplingStrength, dataGravity } = params;

    // Data aliases
    const { x, v, phase, spin, target, hasTarget, regionID, forwardMatrix, activation, delayedActivation, lastActiveTime, hysteresisState } = data;

    let activeTargetCount = 0;
    for(let k = 0; k < count; k++) if(hasTarget[k] === 1) activeTargetCount++;
    const hasActiveTargets = activeTargetCount > 0;

    const effectivePlasticity = teacherFeedback === -1 ? 0.3 : (teacherFeedback === 1 ? 0.0 : plasticity);
    const isEncoding = effectivePlasticity > 0;
    const isRecalling = hasActiveTargets && params.inputText === ""; 

    // Use params.dataGravity for target attraction strength
    // Use fallback values if dataGravity is low (e.g. standard mode default)
    const k_spring_base = dataGravity > 0 ? dataGravity : (isEncoding ? 0.2 : (isRecalling ? 1.5 : (started ? 0.2 : 0.05)));
    const stiffnessMult = isRecalling ? 0.1 : (isEncoding ? 0.1 : 1.0);

    // Spatial Hashing
    spatial.frameCounter++;
    const CELL_SIZE = 5.0;
    const GRID_SIZE = 4096;
    
    if (spatial.frameCounter % 3 === 0) {
        spatial.gridHead.fill(-1);
        spatial.neighborCounts.fill(0);
        
        for (let i = 0; i < count; i++) {
            const xi = Math.floor((x[i*3] + 500) / CELL_SIZE);
            const yi = Math.floor((x[i*3+1] + 500) / CELL_SIZE);
            const zi = Math.floor((x[i*3+2] + 500) / CELL_SIZE);
            const hash = Math.abs((xi * 73856093) ^ (yi * 19349663) ^ (zi * 83492791)) % GRID_SIZE;
            spatial.gridNext[i] = spatial.gridHead[hash];
            spatial.gridHead[hash] = i;
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
                         let j = spatial.gridHead[hash];
                         while (j !== -1 && foundCount < maxNeighbors) {
                             if (j !== i) {
                                 const distSq = (x[j*3]-x[i*3])**2 + (x[j*3+1]-x[i*3+1])**2 + (x[j*3+2]-x[i*3+2])**2;
                                 if (distSq < 25.0) { 
                                     if (offset + foundCount < spatial.neighborList.length) {
                                        spatial.neighborList[offset + foundCount] = j;
                                        foundCount++;
                                     }
                                 }
                             }
                             j = spatial.gridNext[j];
                         }
                    }
                }
            }
            spatial.neighborCounts[i] = foundCount;
        }
    }

    // Output Arrays
    const colorArray = new Float32Array(count * 3);
    const linePositions = [];
    const lineColors = [];
    
    // Stats
    let totalError = 0;
    let totalSpeed = 0;
    let totalKineticEnergy = 0;
    let totalSynapticWeight = 0;
    let sumCosPhase = 0;
    let sumSinPhase = 0;
    let netSpin = 0;

    const delayAlpha = 0.15;
    const continuousStdpRate = 0.05;

    for (let i = 0; i < count; i++) {
        let fx = 0, fy = 0, fz = 0;
        let phaseDelta = 0;
        const idx3 = i * 3;
        const ix = x[idx3], iy = x[idx3 + 1], iz = x[idx3 + 2];
        const rid = regionID[i];
        const nOffset = i * 48;
        const nCount = spatial.neighborCounts[i];
        const isTarget = hasTarget[i] === 1;

        sumCosPhase += Math.cos(phase[i]);
        sumSinPhase += Math.sin(phase[i]);
        netSpin += spin[i];

        // Base Forces
        if (effectiveChaos) {
            fx += (Math.random() - 0.5) * 1.5; fy += (Math.random() - 0.5) * 1.5; fz += (Math.random() - 0.5) * 1.5;
            fx += -iy * 0.05; fy += ix * 0.05;
        } else if (systemTemperature > 0.5) {
            const noise = systemTemperature * 0.25;
            fx += (Math.random() - 0.5) * noise; fy += (Math.random() - 0.5) * noise; fz += (Math.random() - 0.5) * noise;
        } else if (!started) {
            fx += (Math.random() - 0.5) * 0.02; fy += (Math.random() - 0.5) * 0.02; fz += (Math.random() - 0.5) * 0.02;
            fx += -iy * 0.001; fy += ix * 0.001;
        }

        if (hasTarget[i] && !effectiveChaos && started) {
            const dx = target[idx3] - ix; const dy = target[idx3+1] - iy; const dz = target[idx3+2] - iz;
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
            totalError += dist;
            fx += dx * k_spring_base; fy += dy * k_spring_base; fz += dz * k_spring_base;
        }

        const baseInteractionStrength = hasTarget[i] ? 0.1 : 1.0;
        let predictionLock = 0.0;

        if (baseInteractionStrength > 0.01) {
            for (let n = 0; n < nCount; n++) {
                if (nOffset + n >= spatial.neighborList.length) break;
                
                const j = spatial.neighborList[nOffset + n];
                const rj = regionID[j];
                
                if ((rid === 0 && rj === 1) || (rid === 1 && rj === 0)) continue;

                const dx = x[j*3] - ix; const dy = x[j*3+1] - iy; const dz = x[j*3+2] - iz;
                const distSq = dx*dx + dy*dy + dz*dz;
                if (distSq < 0.01 || distSq > 16.0) continue;
                const dist = Math.sqrt(distSq);
                const r0 = equilibriumDistance;
                let force = 0;

                // Paper Physics
                let paperCoupling = 1.0;
                if (usePaperPhysics) {
                    const phaseDiff = phase[i] - phase[j];
                    const phaseTerm = (Math.cos(phaseDiff) + 1.0) / 2.0;
                    const spinTerm = 1.0 + spinCouplingStrength * spin[i] * spin[j];
                    const syncStrength = DyT(phaseTerm, 1.0, 2.0);
                    paperCoupling = (phaseTerm * phaseCouplingStrength) * spinTerm * syncStrength;
                }

                let springF = 0;
                if (dist < r0) springF = -stiffness * stiffnessMult * (r0 - dist) * 2.0;
                else if (!hasTarget[i]) springF = stiffness * stiffnessMult * (dist - r0) * 0.1;

                if (usePaperPhysics) force += springF * paperCoupling;
                else force += springF;

                // Neural Drive
                if (!usePaperPhysics) {
                    // Safety check for forward matrix access
                    const fIdx = j * count + i;
                    if (fIdx < forwardMatrix.length) {
                        const weightJI = forwardMatrix[fIdx];
                        if (weightJI > 0.01) {
                            const signal = weightJI * delayedActivation[j];
                            predictionLock += signal;
                            force += signal * 0.01;
                        }
                    }
                }

                // STDP
                if (isEncoding && !usePaperPhysics) {
                    const hebbianProduct = activation[i] * delayedActivation[j];
                    if (hebbianProduct > 0.1) {
                        const fIdx = j * count + i;
                        if (fIdx < forwardMatrix.length) {
                            const deltaW = hebbianProduct * continuousStdpRate;
                            forwardMatrix[fIdx] += deltaW;
                            if (forwardMatrix[fIdx] > 1.0) forwardMatrix[fIdx] = 1.0;
                            totalSynapticWeight += deltaW;
                        }
                    }
                }

                force *= baseInteractionStrength;
                const invDist = 1.0 / dist;
                fx += dx * invDist * force; fy += dy * invDist * force; fz += dz * invDist * force;

                // Phase Sync
                const phaseDiff = phase[j] - phase[i];
                let syncRate = phaseSyncRate;
                if (usePaperPhysics) syncRate *= (1.0 + spinCouplingStrength * spin[i] * spin[j]);
                phaseDelta += Math.sin(phaseDiff) * 0.1 * syncRate;

                // Lines
                const showLine = usePaperPhysics 
                    ? (paperCoupling > 0.8 && dist < r0 * 2.5) 
                    : (j > i && dist < r0 * 2.0);
                
                if (showLine && linePositions.length < count * 12) { // Cap lines
                    linePositions.push(ix, iy, iz, x[j*3], x[j*3+1], x[j*3+2]);
                    if (usePaperPhysics) {
                        const hue = Math.abs(Math.cos(phaseDiff));
                        lineColors.push(hue * 2, hue, 0.5, hue * 2, hue, 0.5);
                    } else {
                        lineColors.push(0, 0.3, 0.8, 0, 0.3, 0.8);
                    }
                }
            }
        }

        // Hysteresis / Activation
        const externalDrive = hasTarget[i] ? 1.0 : 0.0;
        const totalInputEnergy = externalDrive + predictionLock * 2.0;
        
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
        if (activation[i] > 0.8 && (timeNow - lastActiveTime[i] > 0.5)) lastActiveTime[i] = timeNow;

        // Integration
        let particleDamping = 0.85;
        if (usePaperPhysics) particleDamping = 0.90;
        if (!started) particleDamping = 0.95;
        else if (effectiveChaos) particleDamping = 0.98;

        if (predictionLock > 0.1) particleDamping *= (1.0 - Math.min(0.5, predictionLock * 0.5));

        v[idx3] = v[idx3] * particleDamping + fx;
        v[idx3+1] = v[idx3+1] * particleDamping + fy;
        v[idx3+2] = v[idx3+2] * particleDamping + fz;

        x[idx3] += v[idx3]; x[idx3+1] += v[idx3+1]; x[idx3+2] += v[idx3+2];

        // Bounds check
        const rSq = x[idx3]**2 + x[idx3+1]**2 + x[idx3+2]**2;
        if (rSq > 2500) { x[idx3]*=0.99; x[idx3+1]*=0.99; x[idx3+2]*=0.99; }

        phase[i] += phaseSyncRate * phaseDelta + 0.05;

        // Color Output
        const speedSq = v[idx3]**2 + v[idx3+1]**2 + v[idx3+2]**2;
        const speed = Math.sqrt(speedSq);
        totalSpeed += speed;
        totalKineticEnergy += 0.5 * speedSq;

        const entropy = Math.min(1.0, speed * 0.5); 
        let r=0, g=0, b=0;
        
        if (usePaperPhysics) {
            if (spin[i] > 0) { r = SPIN_UP_COLOR.r; g = SPIN_UP_COLOR.g; b = SPIN_UP_COLOR.b; } 
            else { r = SPIN_DOWN_COLOR.r; g = SPIN_DOWN_COLOR.g; b = SPIN_DOWN_COLOR.b; }
            const pulse = (Math.sin(phase[i]) + 1) * 0.5;
            r += pulse * 0.3; g += pulse * 0.3; b += pulse * 0.3;
        } else {
            if (rid === 0) { r=0.1; g=1.0; b=1.0; }
            else if (rid === 1) { r=1.0; g=0.1; b=1.0; }
            else { r=1.0; g=0.8; b=0.0; }
        }

        const coreMix = entropy * 2.0; 
        TEMP_COLOR.setRGB(r * (1+coreMix), g * (1+coreMix), b * (1+coreMix)); 
        if (activation[i] > 0.5) TEMP_COLOR.lerp(WHITE, activation[i]);

        colorArray[i*3] = TEMP_COLOR.r;
        colorArray[i*3+1] = TEMP_COLOR.g;
        colorArray[i*3+2] = TEMP_COLOR.b;
    }

    const stats: SystemStats = {
        meanError: activeTargetCount > 0 ? totalError / activeTargetCount : 0,
        meanSpeed: totalSpeed / count,
        energy: totalKineticEnergy,
        fps: 0, 
        temperature: systemTemperature,
        isStable: (totalKineticEnergy < 2.0 && totalError < (activeTargetCount * 0.1)) || totalKineticEnergy < 0.05,
        trainingProgress: totalSynapticWeight,
        phaseOrder: Math.sqrt(sumCosPhase * sumCosPhase + sumSinPhase * sumSinPhase) / count,
        spinOrder: Math.abs(netSpin) / count,
        entropy: totalKineticEnergy / count,
        patternMatch: Math.max(0, 1.0 - (totalError / (activeTargetCount || 1)) / 10.0) * 100
    };

    return {
        positions: x,
        colors: colorArray,
        linePositions: new Float32Array(linePositions),
        lineColors: new Float32Array(lineColors),
        stats
    };
}
