import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import ParticleSystem from './ParticleSystem';
import { SimulationParams } from '../types';

// Workaround for missing JSX types in current environment
const Color = 'color' as any;
const AmbientLight = 'ambientLight' as any;
const PointLight = 'pointLight' as any;

interface SimulationCanvasProps {
  params: SimulationParams;
}

const SimulationCanvas: React.FC<SimulationCanvasProps> = ({ params }) => {
  return (
    <div className="w-full h-full relative bg-gray-950">
      <Canvas
        camera={{ position: [0, 0, 30], fov: 45 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <Color attach="background" args={['#050810']} />
        
        {/* Ambient setup */}
        <AmbientLight intensity={0.4} />
        <PointLight position={[10, 10, 10]} intensity={1} />
        <PointLight position={[-10, -10, -10]} intensity={0.5} color="blue" />
        
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        
        {/* Core Simulation */}
        <ParticleSystem params={params} />

        {/* Camera Controls */}
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true} 
          autoRotate={true}
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
};

export default SimulationCanvas;