import React from 'react';

const steps = [
  'Setup',
  'Capture',
  'Scan',
  'Rules Check',
  'Officer Review',
  'Report'
];

interface StepperProps {
  currentStep: number;
}

export default function Stepper({ currentStep }: StepperProps) {
  return (
    <div className="w-full bg-slate-900/50 p-4 border-b border-slate-800 overflow-x-auto">
      <div className="max-w-4xl mx-auto">
        <ul className="flex items-center justify-between min-w-[600px]">
          {steps.map((step, index) => {
            const isCompleted = index < currentStep;
            const isCurrent = index === currentStep;
            const isPending = index > currentStep;
            
            return (
              <li key={step} className="flex flex-col items-center relative flex-1">
                {/* Connecting line */}
                {index !== 0 && (
                  <div 
                    className={`absolute left-0 top-3 -translate-x-1/2 w-full h-[2px] ${
                      isCompleted || isCurrent ? 'bg-amber-500' : 'bg-slate-700'
                    }`} 
                  />
                )}
                
                <div 
                  className={`z-10 w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-colors ${
                    isCompleted 
                      ? 'bg-amber-500 text-slate-950' 
                      : isCurrent 
                        ? 'bg-amber-500 text-slate-950 ring-4 ring-amber-500/20' 
                        : 'bg-slate-800 text-slate-500 border border-slate-700'
                  }`}
                >
                  {isCompleted ? '✓' : index + 1}
                </div>
                <span 
                  className={`mt-2 text-xs font-medium text-center ${
                    isCurrent ? 'text-amber-500' : isCompleted ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  {step}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
