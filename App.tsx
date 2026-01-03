
import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Cylinder, Grid } from '@react-three/drei';
import { EffectComposer, Bloom, ChromaticAberration, Noise, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { SimulationParams, ParticleData, DEFAULT_PARAMS, MemoryAction, CONSTANTS, TestResult, SystemStats, MemorySnapshot } from './types';
import { EXPERIMENT_INFO } from './Info';
import { initSystem, step, updateParams, updateTargets, getMatrixData } from './worker';

// --- ParticleSystem ---
// Workaround for missing JSX types in current environment
const InstancedMesh = 'instancedMesh' as any;
const SphereGeometry = 'sphereGeometry' as any;
const LineSegments = 'lineSegments' as any;
const LineBasicMaterial = 'lineBasicMaterial' as any;
const MeshBasicMaterial = 'meshBasicMaterial' as any;
// Optimized material (Phong is cheaper than Standard PBR)
const MeshPhongMaterial = 'meshPhongMaterial' as any;

const TEMP_OBJ = new THREE.Object3D();

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

interface ParticleSystemProps {
  params: SimulationParams;
  dataRef: React.MutableRefObject<ParticleData>;
  statsRef: React.MutableRefObject<SystemStats>;
  started: boolean;
  teacherFeedback: number; 
  spatialRefs: React.MutableRefObject<{ neighborList: Int32Array; neighborCounts: Int32Array; gridHead: Int32Array; gridNext: Int32Array; frameCounter: number; }>;
}

const ParticleSystem: React.FC<ParticleSystemProps> = ({ params, dataRef, statsRef, started, teacherFeedback, spatialRefs }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const outlineRef = useRef<THREE.InstancedMesh>(null); 
  const ghostRef = useRef<THREE.InstancedMesh>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  
  // Initialize System
  useEffect(() => {
    initSystem(params.particleCount, params, teacherFeedback, started);
  }, [params.particleCount]); // Re-init if count changes

  // Send Updates
  useEffect(() => {
    updateParams(params, started, teacherFeedback);
  }, [params, started, teacherFeedback]);

  // Text Processing
  useEffect(() => {
      if (params.paused || !params.inputText) {
          updateTargets([], []);
          return;
      }
      
      const count = params.particleCount;
      const { positions, count: pointCount } = textToPoints(params.inputText, Math.floor(count * 0.8));
      
      let offsetX = 0;
      if (params.targetRegion === 0) offsetX = -35;
      else if (params.targetRegion === 1) offsetX = 35;

      const targets = [];
      for(let i=0; i<pointCount; i++) {
          targets.push({ x: positions[i*3] + offsetX, y: positions[i*3+1], z: positions[i*3+2] });
      }

      // Shuffle indices
      const indices = Array.from({length: count}, (_, i) => i);
      for (let i = count - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      updateTargets(targets, indices);
  }, [params.inputText, params.targetRegion, params.paused, params.particleCount]);

  // Poll Matrix Data for HUD
  useEffect(() => {
      const interval = setInterval(() => {
          const matrix = getMatrixData();
          if (matrix && matrix.length > 0 && dataRef.current.forwardMatrix.length === matrix.length) {
              dataRef.current.forwardMatrix.set(matrix);
          }
      }, 150);
      return () => clearInterval(interval);
  }, []);

  // Frame Loop
  useFrame((state, delta) => {
      if (params.paused) return;
      const result = step(delta);
      
      if (result) {
        const { positions, colors, linePositions, lineColors, stats } = result;

        // Update Mesh
        if (meshRef.current) {
            const count = positions.length / 3;
            // Check if mesh buffer size matches the physics result size to prevent RangeErrors
            const colorAttr = meshRef.current.instanceColor;
            
            if (colorAttr && colorAttr.array.length === colors.length) {
                for (let i = 0; i < count; i++) {
                    TEMP_OBJ.position.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
                    const s = 0.3; 
                    TEMP_OBJ.scale.set(s, s, s);
                    TEMP_OBJ.updateMatrix();
                    meshRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
                    if (outlineRef.current) outlineRef.current.setMatrixAt(i, TEMP_OBJ.matrix);
                }
                meshRef.current.instanceMatrix.needsUpdate = true;
                if (outlineRef.current) outlineRef.current.instanceMatrix.needsUpdate = true;

                // Colors
                colorAttr.array.set(colors);
                colorAttr.needsUpdate = true;

                if (outlineRef.current && outlineRef.current.instanceColor && outlineRef.current.instanceColor.array.length === colors.length) {
                    outlineRef.current.instanceColor.array.set(colors);
                    outlineRef.current.instanceColor.needsUpdate = true;
                }
            }
        }

        // Update Lines
        if (linesRef.current) {
                linesRef.current.geometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
                linesRef.current.geometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
        }

        // Update Stats
        if (statsRef.current) {
            Object.assign(statsRef.current, stats);
            statsRef.current.fps = 1 / delta; 
        }
      }
  });

  // Initial Geometry
  const maxConnections = params.particleCount * 6;
  const lineGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    return geo;
  }, [maxConnections]);

  return (
    <>
      <InstancedMesh ref={ghostRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[1, 8, 8]} />
        <MeshBasicMaterial color="#ffffff" transparent opacity={0.05} wireframe />
      </InstancedMesh>

      <InstancedMesh ref={outlineRef} args={[undefined, undefined, params.particleCount]}>
        <SphereGeometry args={[0.42, 8, 8]} />
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
        <SphereGeometry args={[0.22, 10, 10]} /> 
        <MeshPhongMaterial 
            color="#050505" 
            specular="#999999"
            shininess={30}
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
    if (!params.showRegions) return null;

    return (
        <group>
            {/* Region A (Cyan) Left */}
            <Cylinder args={[15, 15, 10, 6]} position={[-35, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshBasicMaterial color="#00FFFF" wireframe transparent opacity={0.25} />
            </Cylinder>
            {/* Region B (Pink) Right */}
            <Cylinder args={[15, 15, 10, 6]} position={[35, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshBasicMaterial color="#FF00AA" wireframe transparent opacity={0.25} />
            </Cylinder>
            {/* Region Associative (Gold) Center */}
            <Cylinder args={[12, 12, 30, 8]} position={[0, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshBasicMaterial color="#FFAA00" wireframe transparent opacity={0.15} />
            </Cylinder>
        </group>
    );
};

// --- Matrix HUD ---
const MatrixHUD: React.FC<{ 
    dataRef: React.MutableRefObject<ParticleData>;
    particleCount: number;
}> = ({ dataRef, particleCount }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const ctx = canvasRef.current?.getContext('2d');
        if (!ctx) return;
        
        const updateInterval = setInterval(() => {
            const size = canvasRef.current?.width || 120;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.fillRect(0, 0, size, size);

            // Regions
            const p1 = size * 0.25; const p2 = size * 0.5; const p3 = size;
            ctx.fillStyle = 'rgba(6, 182, 212, 0.05)'; ctx.fillRect(0, 0, p1, p1);
            ctx.fillStyle = 'rgba(236, 72, 153, 0.05)'; ctx.fillRect(p1, p1, p2-p1, p2-p1);
            ctx.fillStyle = 'rgba(234, 179, 8, 0.05)'; ctx.fillRect(p2, p2, p3-p2, p3-p2);

            const scale = size / particleCount;
            const step = Math.max(1, Math.floor(particleCount / 150)); 
            const matrix = dataRef.current.forwardMatrix;
            
            // Check if matrix is ready (prevent flicker on init)
            if (matrix.length === 0) return;

            for(let j=0; j<particleCount; j+=step) {
                const y = Math.floor(j * scale);
                ctx.fillStyle = '#ffffff'; ctx.fillRect(y, y, 1, 1); // Diagonal
                
                for (let i=0; i<particleCount; i+=step) {
                    const idx = j * particleCount + i;
                    if (idx < matrix.length) {
                        const wJI = matrix[idx];
                        if (wJI > 0.05) {
                            const x = Math.floor(i * scale);
                            ctx.fillStyle = `rgba(74, 222, 128, ${Math.min(1.0, wJI * 3.0)})`;
                            ctx.fillRect(x, y, 1, 1);
                        }
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

// --- Active Inference Control Panel ---
const InferenceControlPanel: React.FC<{ 
    setFeedback: (val: number) => void,
    feedback: number
}> = ({ setFeedback, feedback }) => {
    return (
        <div className="absolute top-1/2 left-6 -translate-y-1/2 flex flex-col gap-4 pointer-events-auto z-50">
             <div className="text-[10px] text-yellow-400 font-bold uppercase tracking-widest mb-[-10px] ml-1">Thermodynamic Controls</div>
             <div className="flex flex-col gap-2 p-2 bg-black/80 border border-yellow-800 rounded backdrop-blur-md w-24 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-full bg-yellow-900/10 pointer-events-none animate-pulse z-0"></div>
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10 opacity-30">
                    <span className="text-[40px] text-yellow-500 font-bold rotate-[-45deg] border-2 border-yellow-500 px-2 rounded">AUTO</span>
                </div>

                <button 
                    onMouseDown={() => setFeedback(1)} 
                    onMouseUp={() => setFeedback(0)}
                    onMouseLeave={() => setFeedback(0)}
                    className={`h-16 border-2 transition-all duration-100 flex flex-col items-center justify-center font-bold text-[10px] tracking-widest relative z-20 ${
                        feedback === 1 
                        ? 'border-green-400 bg-green-500/20 text-green-400 shadow-[0_0_20px_rgba(34,197,94,0.5)] scale-95' 
                        : 'border-green-900/50 text-green-800 hover:border-green-500/50 hover:text-green-500'
                    }`}
                >
                    <span className="text-xl mb-1">❄️</span>
                    <span>FREEZE</span>
                </button>
                
                <div className="h-px bg-gray-800 w-full relative z-20"></div>
                
                <button 
                    onMouseDown={() => setFeedback(-1)} 
                    onMouseUp={() => setFeedback(0)}
                    onMouseLeave={() => setFeedback(0)}
                    className={`h-16 border-2 transition-all duration-100 flex flex-col items-center justify-center font-bold text-[10px] tracking-widest relative z-20 ${
                        feedback === -1 
                        ? 'border-red-400 bg-red-500/20 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.5)] scale-95' 
                        : 'border-red-900/50 text-red-800 hover:border-red-500/50 hover:text-red-500'
                    }`}
                >
                    <span className="text-xl mb-1">🔥</span>
                    <span>AGITATE</span>
                </button>
             </div>
             
             <div className="h-32 w-4 bg-gray-900 border border-gray-800 relative ml-10">
                 <div 
                    className={`absolute bottom-0 w-full transition-all duration-300 ${feedback === -1 ? 'bg-red-500 h-full' : (feedback === 1 ? 'bg-green-500 h-[5%]' : 'bg-cyan-500 h-[20%]')}`}
                 ></div>
                 <div className="absolute -left-6 bottom-0 text-[9px] text-gray-500 rotate-[-90deg] origin-bottom-right translate-x-full mb-1">TEMP</div>
             </div>
             
             <div className="bg-black/80 border border-yellow-800/50 p-1 text-[9px] text-yellow-300 font-mono text-center tracking-wider animate-pulse">
                 AUTO-AGENCY ACTIVE
             </div>
        </div>
    )
}

// --- Active Inference Auto-Controller (Procedural) ---
const ActiveInferenceController: React.FC<{
    statsRef: React.MutableRefObject<SystemStats>;
    setTeacherFeedback: (val: number) => void;
}> = ({ statsRef, setTeacherFeedback }) => {
    useEffect(() => {
        const interval = setInterval(() => {
            if (!statsRef.current) return;
            const { energy, patternMatch } = statsRef.current;
            
            if (patternMatch > 90) {
                setTeacherFeedback(1); // FREEZE
            } else if (energy < 0.8 && patternMatch < 60) {
                setTeacherFeedback(-1); // AGITATE
                setTimeout(() => setTeacherFeedback(0), 300); // Short burst
            } else {
                setTeacherFeedback(0); // NEUTRAL
            }
        }, 500); 

        return () => {
            clearInterval(interval);
            setTeacherFeedback(0);
        };
    }, []);

    return null;
}

// --- Temporal Training Controller (State Machine) ---
interface TelemetryFrame {
    time: number;
    phase: string;
    energy: number;
    error: number;
    synapticWeight: number;
}

interface TemporalControllerProps {
    enabled: boolean;
    state: 'idle' | 'training' | 'ready';
    setState: (s: 'idle' | 'training' | 'ready') => void;
    setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
    statsRef: React.MutableRefObject<SystemStats>;
    addLog: (s: string) => void;
    telemetryRef: React.MutableRefObject<TelemetryFrame[]>;
}

const TemporalTrainingController: React.FC<TemporalControllerProps> = ({ enabled, state, setState, setParams, statsRef, addLog, telemetryRef }) => {
    const trainingStage = useRef(0);
    const stepTimer = useRef(0);
    const waitingForStable = useRef(false);

    useEffect(() => {
        if (!enabled || state !== 'training') {
            trainingStage.current = 0;
            return;
        }
        
        // Reset telemetry on start
        telemetryRef.current = [];

        const interval = setInterval(() => {
            if (!statsRef.current) return;
            const { isStable, energy, meanError, trainingProgress } = statsRef.current;
            stepTimer.current += 100;
            
            const cycle = Math.floor(trainingStage.current / 2);
            const phaseType = trainingStage.current % 2 === 0 ? "TICK" : "TOCK";

            // RECORD TELEMETRY
            telemetryRef.current.push({
                time: Date.now(),
                phase: phaseType,
                energy: energy,
                error: meanError,
                synapticWeight: trainingProgress
            });

            if (cycle >= 3) {
                 // FINISHED
                 setParams(p => ({ ...p, inputText: "", plasticity: 0 }));
                 setState('ready');
                 addLog("TRAINING COMPLETE. READY FOR PREDICTION.");
                 return;
            }

            if (waitingForStable.current) {
                // Wait for physics to settle before advancing
                if (isStable && stepTimer.current > 1500) {
                     waitingForStable.current = false;
                     trainingStage.current++;
                     stepTimer.current = 0;
                     // Trigger next phase immediately
                } else {
                    return; // Keep waiting
                }
            }

            if (!waitingForStable.current) {
                // START NEW PHASE
                const phase = trainingStage.current % 2; 
                if (phase === 0) {
                     // TICK PHASE
                     addLog(`CYCLE ${cycle + 1}: IMPRINTING 'TICK' (A)...`);
                     setParams(p => ({ ...p, inputText: "TICK", targetRegion: 0, plasticity: 0.2 }));
                     waitingForStable.current = true;
                } else {
                     // TOCK PHASE
                     addLog(`CYCLE ${cycle + 1}: ASSOCIATING 'TOCK' (B)...`);
                     // Keep TICK visible (persistence) so correlation can happen, add TOCK
                     setParams(p => ({ ...p, inputText: "TOCK", targetRegion: 1, plasticity: 0.2 }));
                     waitingForStable.current = true;
                }
            }

        }, 100);

        return () => clearInterval(interval);
    }, [enabled, state]);

    return null;
};

// --- Info Modal ---
const InfoModal: React.FC<{ 
    mode: string, 
    onClose: () => void 
}> = ({ mode, onClose }) => {
    const info = EXPERIMENT_INFO[mode] || EXPERIMENT_INFO['standard'];

    return (
        <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/60 backdrop-blur-md p-6">
            <div className="max-w-xl w-full bg-black/90 border border-cyan-500/50 p-6 shadow-[0_0_50px_rgba(6,182,212,0.2)] relative overflow-hidden">
                {/* Decorative Elements */}
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-500"></div>
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-500"></div>
                <div className="scanlines opacity-20"></div>

                <div className="relative z-10">
                    <div className="flex justify-between items-start mb-4 border-b border-gray-800 pb-2">
                        <h2 className="text-2xl font-bold text-cyan-400 tracking-widest uppercase">{info.title}</h2>
                        <button onClick={onClose} className="text-cyan-700 hover:text-cyan-300 font-bold">[ CLOSE ]</button>
                    </div>

                    <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
                        <div>
                            <h3 className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-1">Summary</h3>
                            <p className="text-sm text-gray-300 leading-relaxed">{info.summary}</p>
                        </div>

                        <div>
                            <h3 className="text-xs text-green-700 font-bold uppercase tracking-wider mb-1">Why It Matters</h3>
                            <p className="text-sm text-gray-400 italic border-l-2 border-green-900/50 pl-3">{info.importance}</p>
                        </div>

                        <div>
                            <h3 className="text-xs text-yellow-700 font-bold uppercase tracking-wider mb-1">Demonstration</h3>
                            <p className="text-sm text-gray-300">{info.demonstration}</p>
                        </div>

                        <div className="bg-cyan-950/20 p-3 border border-cyan-900/30">
                            <h3 className="text-xs text-cyan-600 font-bold uppercase tracking-wider mb-2">Instructions</h3>
                            <ol className="list-decimal list-inside space-y-2">
                                {info.steps.map((step, i) => (
                                    <li key={i} className="text-xs text-cyan-100 font-mono">{step}</li>
                                ))}
                            </ol>
                        </div>
                    </div>

                    <button onClick={onClose} className="w-full mt-6 py-3 bg-cyan-900/30 hover:bg-cyan-700/50 text-cyan-400 font-bold tracking-widest border border-cyan-600/30 transition-all text-xs">
                        RESUME SIMULATION
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- UI Overlay ---

interface UIOverlayProps {
  params: SimulationParams;
  setParams: React.Dispatch<React.SetStateAction<SimulationParams>>;
  dataRef: React.MutableRefObject<ParticleData>;
  simulationMode: 'standard' | 'temporal' | 'inference' | 'paper';
  statsRef: React.MutableRefObject<SystemStats>;
  onTestComplete: (results: TestResult[]) => void;
  telemetryRef: React.MutableRefObject<TelemetryFrame[]>;
  onShowInfo: () => void;
  onExit: () => void;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ params, setParams, dataRef, simulationMode, statsRef, onTestComplete, telemetryRef, onShowInfo, onExit }) => {
  const [autoRunPhase, setAutoRunPhase] = useState<'idle' | 'reset' | 'entropy' | 'observation' | 'encoding' | 'amnesia' | 'recall'>('idle');
  const [testLogs, setTestLogs] = useState<string[]>([]);
  const [temporalState, setTemporalState] = useState<'idle' | 'training' | 'ready'>('idle');
  const [isMinimized, setIsMinimized] = useState(false);

  const handleChange = (key: keyof SimulationParams, value: number | string | object | boolean) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const togglePause = () => handleChange('paused', !params.paused);
  
  const handleMemoryAction = (type: 'save' | 'load', slot: number) => {
    setParams(prev => ({ ...prev, memoryAction: { type, slot, triggerId: prev.memoryAction.triggerId + 1 } }));
  };

  const addLog = (msg: string) => {
      setTestLogs(prev => [...prev.slice(-4), `> ${msg}`]);
  };
  
  const handleExport = () => {
      const dataStr = JSON.stringify(telemetryRef.current, null, 2);
      const blob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `experiment_b_telemetry_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      addLog("DATA EXPORTED SUCCESSFULLY.");
  };

  // --- PAPER DEMO SETUP ---
  useEffect(() => {
      if (simulationMode === 'paper') {
          // Initialize for L-Group Demo
          setParams(prev => ({ 
              ...prev, 
              usePaperPhysics: true, 
              showRegions: false, 
              inputText: "", 
              particleCount: 1500, // Higher count for wave effects
              couplingDecay: 6.0,
              spinCouplingStrength: 0.5,
              phaseCouplingStrength: 1.0,
              chaosMode: false
          }));
          addLog("L-GROUP FRAMEWORK INITIALIZED.");
          addLog("SPIN STATES: RED (+1/2), BLUE (-1/2)");
      }
  }, [simulationMode]);

  // --- EXPERIMENT A: ACTIVE INFERENCE (THERMODYNAMICS) ---
  useEffect(() => {
      if (simulationMode === 'inference') {
          // Changed targetRegion from 0 (Left) to 2 (Associative/Center)
          setParams(prev => ({ ...prev, showRegions: true, inputText: "ORDER", targetRegion: 2 }));
          addLog("SYSTEM INITIALIZED. PATTERN: 'ORDER'");
          addLog("AGENCY: PROCEDURAL");
      }
  }, [simulationMode]);

  // --- EXPERIMENT B: TEMPORAL PREDICTION ---
  const runTemporalTraining = () => {
      setTemporalState('training');
      // Logic handled by TemporalTrainingController
  };

  const triggerTemporalCue = () => {
      addLog("CUE: TRIGGERING 'TICK'...");
      setParams(p => ({ ...p, inputText: "TICK", targetRegion: 0 }));
      setTimeout(() => {
          setParams(p => ({ ...p, inputText: "" }));
          addLog("OBSERVE: 'TOCK' GHOST IN REGION B");
      }, 2000);
  };

  const togglePlasticity = (active: boolean) => {
    if (active) setParams(prev => ({ ...prev, plasticity: 0.1, dataGravity: Math.max(prev.dataGravity, 0.5) }));
    else setParams(prev => ({ ...prev, plasticity: 0 }));
  };

  // Step-by-Step Experiment Controller
  useEffect(() => {
    if (simulationMode !== 'standard') return; 
    if (autoRunPhase === 'idle') return;

    if (autoRunPhase === 'reset') setParams(DEFAULT_PARAMS);
    else if (autoRunPhase === 'entropy') setParams(prev => ({ ...prev, chaosMode: true, inputText: "" }));
    else if (autoRunPhase === 'observation') setParams(prev => ({ ...prev, chaosMode: false, inputText: "QUANTUM", dataGravity: 0.4 }));
    else if (autoRunPhase === 'encoding') { togglePlasticity(true); setTimeout(() => handleMemoryAction('save', 1), 1000); }
    else if (autoRunPhase === 'amnesia') { togglePlasticity(false); setParams(prev => ({ ...prev, inputText: "", chaosMode: true })); }
    else if (autoRunPhase === 'recall') { setParams(prev => ({ ...prev, chaosMode: false })); handleMemoryAction('load', 1); }
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

  const panelClass = "bg-black/80 backdrop-blur-md border border-cyan-500/30 p-3 text-cyan-100 shadow-[0_0_15px_rgba(6,182,212,0.15)] relative overflow-hidden transition-all duration-300 ease-in-out";

  return (
    <div className="absolute top-0 right-0 p-4 w-full md:w-80 h-full pointer-events-none flex flex-col items-end font-['Rajdhani'] pb-8">
      <div className="scanlines"></div>

      <div className={`${panelClass} pointer-events-auto w-full clip-path-polygon ${isMinimized ? 'h-[60px]' : 'h-auto'} flex flex-col gap-3`}>
        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-400"></div>
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-400"></div>

        <div className="flex justify-between items-center border-b border-cyan-900/50 pb-2 flex-shrink-0">
            <h1 className="text-lg font-bold text-cyan-400 tracking-widest drop-shadow-[0_0_5px_rgba(34,211,238,0.8)]">NEURO_HOLO</h1>
            <div className="flex items-center gap-1">
                 <button 
                    onClick={togglePause} 
                    className={`p-1.5 border border-cyan-500/50 px-2.5 transition-colors ${params.paused ? 'bg-red-500/50 text-white animate-pulse border-red-400' : 'text-cyan-600 hover:text-cyan-300'}`}
                    title="Pause/Resume"
                 >
                    {params.paused ? "||" : ">>"}
                 </button>
                 <button 
                    onClick={onShowInfo} 
                    className="p-1.5 border border-cyan-500/50 text-cyan-600 hover:text-cyan-300 font-bold px-2.5"
                    title="Info & Instructions"
                 >
                     ?
                 </button>
                 <button 
                    onClick={() => setIsMinimized(!isMinimized)} 
                    className="p-1.5 border border-cyan-500/50 text-cyan-600 hover:text-cyan-300 font-bold px-2.5"
                    title={isMinimized ? "Expand" : "Minimize"}
                 >
                     {isMinimized ? "+" : "_"}
                 </button>
                 <button 
                    onClick={onExit} 
                    className="p-1.5 border border-red-500/50 text-red-600 hover:bg-red-900/30 hover:text-red-400 font-bold px-2.5"
                    title="Exit to Title"
                 >
                     X
                 </button>
            </div>
        </div>

        {/* CONTROLLER PANEL */}
        <div className={`space-y-3 pt-2 overflow-hidden transition-opacity duration-300 ${isMinimized ? 'opacity-0' : 'opacity-100'}`}>
            <div className="bg-cyan-950/30 border border-cyan-500/20 p-3 text-center relative">
                {simulationMode === 'paper' ? (
                    <div className="text-left">
                        <div className="flex items-center gap-2 mb-2 border-b border-purple-800 pb-2">
                            <div className="w-2 h-2 bg-purple-500 animate-pulse"></div>
                            <span className="text-xs text-purple-300 font-bold tracking-widest">EXP C: L-GROUP DYNAMICS</span>
                        </div>
                        <div className="text-[10px] text-purple-100 font-mono mb-2">
                             Visualizing vibrational coupling modulated by Intrinsic Spin and Phase (Eq. 30).
                        </div>
                        
                        <div className="flex flex-col gap-2">
                             <div className="flex items-center justify-between">
                                 <span className="text-[9px] text-purple-300">SPIN COUPLING (γ)</span>
                                 <input 
                                    type="range" min="0" max="1" step="0.1" 
                                    value={params.spinCouplingStrength} 
                                    onChange={(e) => handleChange('spinCouplingStrength', parseFloat(e.target.value))}
                                    className="w-20"
                                 />
                             </div>
                             <div className="flex items-center justify-between">
                                 <span className="text-[9px] text-purple-300">PHASE COUPLING (α)</span>
                                 <input 
                                    type="range" min="0" max="2" step="0.1" 
                                    value={params.phaseCouplingStrength} 
                                    onChange={(e) => handleChange('phaseCouplingStrength', parseFloat(e.target.value))}
                                    className="w-20"
                                 />
                             </div>
                             
                             <button onClick={() => handleChange('inputText', params.inputText ? "" : "L-GROUP")} className="mt-2 py-1 px-2 border border-purple-500/50 text-[9px] hover:bg-purple-900/50">
                                 TOGGLE STRUCTURE INPUT
                             </button>
                        </div>
                    </div>
                ) : simulationMode === 'inference' ? (
                     <div className="text-left">
                        <div className="flex items-center gap-2 mb-2 border-b border-yellow-800 pb-2">
                            <div className="w-2 h-2 bg-yellow-500 animate-pulse"></div>
                            <span className="text-xs text-yellow-300 font-bold tracking-widest">EXP A: ACTIVE INFERENCE</span>
                        </div>
                        <div className="text-[10px] text-yellow-200 font-mono opacity-80 mb-2 leading-relaxed">
                            <span className="text-white font-bold">GOAL:</span> Use Thermodynamic controls (Left) to stabilize the pattern "ORDER".
                        </div>
                    </div>
                ) : simulationMode === 'temporal' ? (
                     <div className="text-left">
                        <div className="flex items-center gap-2 mb-2 border-b border-green-800 pb-2">
                            <div className="w-2 h-2 bg-green-500 animate-pulse"></div>
                            <span className="text-xs text-green-300 font-bold tracking-widest">EXP B: CAUSAL PREDICTION</span>
                        </div>
                        
                        <TemporalTrainingController 
                            enabled={simulationMode === 'temporal'} 
                            state={temporalState} 
                            setState={setTemporalState}
                            setParams={setParams}
                            statsRef={statsRef}
                            addLog={addLog}
                            telemetryRef={telemetryRef}
                        />

                        <div className="flex flex-col gap-2 mt-2">
                            <button 
                                onClick={runTemporalTraining} 
                                disabled={temporalState === 'training'}
                                className={`w-full py-2 bg-green-900/40 border border-green-500/50 text-green-300 font-bold text-xs tracking-widest hover:bg-green-800/60 ${temporalState === 'training' ? 'opacity-50 cursor-wait' : ''}`}
                            >
                                {temporalState === 'training' ? 'TRAINING (DYNAMIC)...' : '1. IMPRINT: TICK -> TOCK'}
                            </button>

                            <button 
                                onClick={triggerTemporalCue} 
                                disabled={temporalState !== 'ready'}
                                className={`w-full py-2 bg-white/10 border border-white/30 text-white font-bold text-xs tracking-widest hover:bg-white/20 ${temporalState !== 'ready' ? 'opacity-30' : ''}`}
                            >
                                2. TRIGGER: "TICK" ONLY
                            </button>
                            
                            {temporalState === 'ready' && (
                                <button onClick={handleExport} className="w-full py-2 bg-green-900/40 border border-green-500 text-white font-bold text-xs tracking-widest hover:bg-green-800/60">
                                    EXPORT DATA [JSON]
                                </button>
                            )}
                            
                            {temporalState === 'training' && (
                                <div className="text-[9px] text-green-400 font-mono text-center animate-pulse">
                                    AWAITING CONVERGENCE...
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    /* STANDARD MODE CONTROLS */
                        <div className="text-left">
                             {autoRunPhase === 'idle' ? (
                                <button onClick={() => setAutoRunPhase('reset')} className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-sm tracking-widest shadow-[0_0_20px_rgba(6,182,212,0.6)] animate-pulse clip-corner transition-all">
                                    INITIALIZE EXPERIMENT
                                </button>
                             ) : (
                                 <>
                                     <div className="flex items-center gap-2 mb-2 border-b border-cyan-800 pb-2">
                                        <div className="w-2 h-2 bg-green-400 animate-pulse"></div>
                                        <span className="text-xs text-cyan-300 font-bold tracking-widest">PHASE: {autoRunPhase.toUpperCase()}</span>
                                     </div>
                                     <p className="text-xs text-white font-mono bg-black/50 p-2 border-l-2 border-cyan-500 mb-3 min-h-[40px]">{getStandardStatusText()}</p>
                                     <button onClick={nextStep} className="w-full py-2 bg-cyan-700 hover:bg-cyan-500 text-white font-bold text-xs tracking-widest border border-cyan-400/50 transition-all flex justify-between px-4 items-center group">
                                        <span>{getNextButtonText()}</span>
                                        <span className="group-hover:translate-x-1 transition-transform">>></span>
                                     </button>
                                 </>
                             )}
                        </div>
                )}
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
      </div>
    </div>
  );
}

const StatusBar: React.FC<{ 
    statsRef: React.MutableRefObject<SystemStats>;
    params: SimulationParams;
    mode: string;
}> = ({ statsRef, params, mode }) => {
    const [displayStats, setDisplayStats] = useState<SystemStats | null>(null);

    useEffect(() => {
        const interval = setInterval(() => {
            if (statsRef.current) {
                setDisplayStats({ ...statsRef.current });
            }
        }, 200); // 5Hz update
        return () => clearInterval(interval);
    }, []);

    if (!displayStats) return null;

    return (
        <div className="absolute top-4 left-4 z-50 pointer-events-none select-none">
            <div className="bg-black/80 backdrop-blur-md border border-cyan-500/30 p-2 shadow-[0_0_20px_rgba(6,182,212,0.15)] w-64 font-mono text-[10px] text-cyan-300">
                 <div className="flex justify-between items-center border-b border-cyan-800 pb-1 mb-2">
                    <span className="font-bold tracking-widest text-cyan-100">SYS_TELEMETRY // {mode.toUpperCase()}</span>
                    <div className={`w-2 h-2 rounded-full ${displayStats.isStable ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-red-500 animate-pulse'}`}></div>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-y-1 gap-x-4">
                     <div className="flex justify-between"><span>FPS</span> <span className="text-white">{Math.round(displayStats.fps)}</span></div>
                     <div className="flex justify-between"><span>COUNT</span> <span className="text-white">{params.particleCount}</span></div>
                     <div className="flex justify-between"><span>ENERGY</span> <span className={`${displayStats.energy > 1.0 ? 'text-red-400' : 'text-white'}`}>{displayStats.energy.toFixed(2)}</span></div>
                     <div className="flex justify-between"><span>TEMP</span> <span className="text-white">{displayStats.temperature.toFixed(2)}</span></div>
                 </div>

                 <div className="mt-2 space-y-1">
                     <div className="flex items-center gap-2">
                         <span className="w-10 text-cyan-600">ENTROPY</span>
                         <div className="flex-1 bg-gray-900 h-1">
                             <div className="h-full bg-cyan-600 transition-all duration-300" style={{width: `${Math.min(100, displayStats.entropy * 100)}%`}}></div>
                         </div>
                     </div>
                     <div className="flex items-center gap-2">
                         <span className="w-10 text-purple-500">MATCH</span>
                         <div className="flex-1 bg-gray-900 h-1">
                             <div className="h-full bg-purple-500 transition-all duration-300" style={{width: `${displayStats.patternMatch}%`}}></div>
                         </div>
                     </div>
                     {mode === 'paper' && (
                         <div className="flex items-center gap-2">
                            <span className="w-10 text-yellow-500">SYNC</span>
                            <div className="flex-1 bg-gray-900 h-1">
                                <div className="h-full bg-yellow-500 transition-all duration-300" style={{width: `${displayStats.phaseOrder * 100}%`}}></div>
                            </div>
                         </div>
                     )}
                 </div>
            </div>
        </div>
    );
};

const App = () => {
    const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
    const [started, setStarted] = useState(false); // Controls if we are in sim or menu
    const [simulationMode, setSimulationMode] = useState<'standard' | 'temporal' | 'inference' | 'paper'>('standard');
    const [teacherFeedback, setTeacherFeedback] = useState(0);
    const [showInfo, setShowInfo] = useState(false);

    // Refs
    const dataRef = useRef<ParticleData>({
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
    });
    
    const statsRef = useRef<SystemStats>({
        meanError: 0,
        meanSpeed: 0,
        energy: 0,
        fps: 0,
        temperature: 0,
        isStable: false,
        trainingProgress: 0,
        phaseOrder: 0,
        spinOrder: 0,
        entropy: 0,
        patternMatch: 0
    });
    
    const spatialRefs = useRef({
        neighborList: new Int32Array(0),
        neighborCounts: new Int32Array(0),
        gridHead: new Int32Array(0),
        gridNext: new Int32Array(0),
        frameCounter: 0
    });
    
    const telemetryRef = useRef<any[]>([]);

    // Menu Screen
    if (!started) {
        return (
            <div className="w-full h-screen bg-black text-cyan-500 font-['Rajdhani'] flex flex-col items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 scanlines opacity-30 pointer-events-none"></div>
                <div className="z-10 flex flex-col items-center gap-8 max-w-2xl w-full p-8 bg-black/80 border border-cyan-500/30 backdrop-blur-md shadow-[0_0_50px_rgba(6,182,212,0.1)]">
                    <div className="text-center">
                        <h1 className="text-6xl font-bold tracking-[0.2em] mb-2 text-transparent bg-clip-text bg-gradient-to-b from-cyan-300 to-cyan-700 drop-shadow-[0_0_10px_rgba(6,182,212,0.8)]">NEURO_HOLO</h1>
                        <p className="text-sm tracking-[0.5em] text-cyan-600 uppercase">Predictive Morphology Engine v2.0</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                         {[
                             { id: 'standard', name: 'STANDARD MODEL', desc: 'Baseline Physics & Memory' },
                             { id: 'inference', name: 'ACTIVE INFERENCE', desc: 'Thermodynamic Agency (Exp A)' },
                             { id: 'temporal', name: 'TEMPORAL LEARNING', desc: 'Causal Prediction (Exp B)' },
                             { id: 'paper', name: 'L-GROUP DYNAMICS', desc: 'Harmonic Spin Coupling (Exp C)' }
                         ].map(mode => (
                             <button 
                                key={mode.id}
                                onClick={() => {
                                    setSimulationMode(mode.id as any);
                                    setStarted(true);
                                    setShowInfo(true);
                                }}
                                className="group relative p-6 border border-cyan-800 hover:border-cyan-400 bg-cyan-950/10 hover:bg-cyan-900/30 transition-all text-left overflow-hidden"
                             >
                                 <div className="absolute top-0 left-0 w-1 h-full bg-cyan-600 transform scale-y-0 group-hover:scale-y-100 transition-transform"></div>
                                 <h3 className="text-xl font-bold tracking-widest text-cyan-300 group-hover:text-white mb-1">{mode.name}</h3>
                                 <p className="text-xs text-cyan-700 group-hover:text-cyan-500 font-mono">{mode.desc}</p>
                             </button>
                         ))}
                    </div>

                    <div className="w-full h-px bg-cyan-900/50"></div>
                    <p className="text-[10px] text-cyan-800 font-mono">
                        SYSTEM STATUS: READY // GPU ACCELERATION: ACTIVE
                    </p>
                </div>
            </div>
        );
    }

    return (
      <div className="w-full h-screen bg-black overflow-hidden relative selection:bg-cyan-500/30 font-sans">
        <Canvas camera={{ position: [0, 0, 120], fov: 35 }} gl={{ antialias: false, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.5 }} dpr={[1, 2]}>
            <color attach="background" args={['#020202']} />
            <fog attach="fog" args={['#020202', 80, 250]} />
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            <ambientLight intensity={0.2} />
            <pointLight position={[10, 10, 10]} intensity={0.5} />
            
            <ParticleSystem 
                params={params} 
                dataRef={dataRef} 
                statsRef={statsRef} 
                started={started} 
                teacherFeedback={teacherFeedback}
                spatialRefs={spatialRefs}
            />
            
            <RegionGuides params={params} />
            
            <OrbitControls makeDefault maxDistance={200} minDistance={20} />
            
            <EffectComposer enableNormalPass={false}>
                <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} radius={0.4} />
                <ChromaticAberration offset={new THREE.Vector2(0.002, 0.002)} radialModulation={false} modulationOffset={0} />
                <Noise opacity={0.05} />
                <Vignette eskil={false} offset={0.1} darkness={1.1} />
            </EffectComposer>
        </Canvas>

        {simulationMode === 'inference' && <InferenceControlPanel setFeedback={setTeacherFeedback} feedback={teacherFeedback} />}
        {simulationMode === 'inference' && <ActiveInferenceController statsRef={statsRef} setTeacherFeedback={setTeacherFeedback} />}
        
        <StatusBar statsRef={statsRef} params={params} mode={simulationMode} />
        <MatrixHUD dataRef={dataRef} particleCount={params.particleCount} />
        
        <UIOverlay 
            params={params} 
            setParams={setParams} 
            dataRef={dataRef} 
            simulationMode={simulationMode}
            statsRef={statsRef}
            onTestComplete={(results) => console.log(results)}
            telemetryRef={telemetryRef}
            onShowInfo={() => setShowInfo(true)}
            onExit={() => { setStarted(false); setParams(DEFAULT_PARAMS); }}
        />
        
        {showInfo && <InfoModal mode={simulationMode} onClose={() => setShowInfo(false)} />}
      </div>
    );
};

export default App;
