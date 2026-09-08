'use client';

import { useState } from 'react';
import Stepper from './Stepper';

export default function InspectionWorkflow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadedImages, setUploadedImages] = useState<{id: string, url: string, number: number}[]>([]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newImages = Array.from(e.target.files).map((file, idx) => ({
        id: Math.random().toString(36).substring(2, 9),
        url: URL.createObjectURL(file),
        number: uploadedImages.length + idx + 4 // Starts from 4 since there are 3 placeholders
      }));
      setUploadedImages([...uploadedImages, ...newImages]);
    }
    // reset input value so the same file can be selected again if needed
    e.target.value = '';
  };

  const nextStep = () => setCurrentStep((prev) => Math.min(prev + 1, 5));
  const prevStep = () => setCurrentStep((prev) => Math.max(prev - 1, 0));

  const renderStepContent = () => {
    switch (currentStep) {
      case 0: // Setup
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-slate-100">Inspection Setup</h2>
            <p className="text-slate-400 text-sm">Verify the auto-captured context before proceeding.</p>
            
            <div className="space-y-4 mt-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                  <label className="block text-[11px] font-medium text-slate-500 uppercase tracking-wider mb-1">Inspection ID</label>
                  <div className="text-sm font-medium text-slate-200">LM-2026-00047</div>
                </div>
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                  <label className="block text-[11px] font-medium text-slate-500 uppercase tracking-wider mb-1">Date & Time</label>
                  <div className="text-sm font-medium text-slate-200 suppress-hydration-warning">
                    {new Date().toLocaleString('en-IN', { 
                      day: '2-digit', month: 'short', year: 'numeric',
                      hour: '2-digit', minute: '2-digit'
                    })}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                <label className="block text-[11px] font-medium text-slate-500 uppercase tracking-wider mb-1">Inspector</label>
                <div className="text-sm font-medium text-slate-200 flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-[9px] text-slate-400">LM</div>
                  Inspector &mdash; LM Officer 0142
                </div>
              </div>

              <div className="bg-amber-950/20 p-4 rounded-lg border border-amber-900/30 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-amber-600/50"></div>
                <div className="flex justify-between items-start mb-3">
                  <label className="block text-xs font-semibold text-amber-500/90 uppercase tracking-wider">System-Provided Context</label>
                  <span className="flex h-2 w-2 relative" title="Live location/system active">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                  </span>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <label className="block text-[11px] text-slate-500 mb-0.5">Inspection Location</label>
                    <div className="text-sm font-medium text-slate-300">Hyderabad, Telangana</div>
                  </div>
                  <div>
                    <label className="block text-[11px] text-slate-500 mb-0.5">Entity / Premises Name</label>
                    <div className="text-sm font-medium text-slate-300">ABC Supermarket</div>
                  </div>
                  <div>
                    <label className="block text-[11px] text-slate-500 mb-0.5">Registered Address</label>
                    <div className="text-sm text-slate-400">Hyderabad, Telangana</div>
                  </div>
                </div>
              </div>

              <div className="pt-2">
                <label className="block text-sm font-medium text-slate-300 mb-2">Inspection Type</label>
                <select className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-amber-500">
                  <option>Routine Inspection</option>
                  <option>Consumer Complaint</option>
                  <option>Surprise Inspection</option>
                  <option>Follow-up Inspection</option>
                </select>
              </div>
            </div>
          </div>
        );
      case 1: // Capture
        return (
          <div className="space-y-4">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-semibold text-slate-100">Capture Product Evidence</h2>
                <p className="text-slate-400 text-sm mt-1">Capture clear images of the package panels containing declarations. Multiple images may be required.</p>
              </div>
            </div>
            
            <div className="bg-amber-900/10 border border-amber-900/30 p-3 rounded-lg flex items-start gap-3 mt-4">
              <svg className="w-5 h-5 text-amber-500 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <p className="text-sm text-amber-200/80">Ensure declarations are clearly visible and readable.</p>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6">
              {/* Card 1 */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg overflow-hidden group">
                <div className="h-32 bg-slate-800 flex items-center justify-center relative">
                  <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                  <div className="absolute inset-0 bg-slate-950/50 hidden group-hover:flex items-center justify-center transition-all cursor-pointer">
                    <span className="text-xs font-medium text-white bg-black/60 px-2 py-1 rounded">Retake Image</span>
                  </div>
                </div>
                <div className="p-3">
                  <div className="flex justify-between items-start mb-1">
                    <h3 className="text-sm font-medium text-slate-200">Image 1</h3>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-green-900/30 text-green-400 border border-green-800">Captured</span>
                  </div>
                  <p className="text-xs text-slate-400">Front / Declaration Panel</p>
                </div>
              </div>

              {/* Card 2 */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg overflow-hidden group">
                <div className="h-32 bg-slate-800 flex items-center justify-center relative">
                  <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                  <div className="absolute inset-0 bg-slate-950/50 hidden group-hover:flex items-center justify-center transition-all cursor-pointer">
                    <span className="text-xs font-medium text-white bg-black/60 px-2 py-1 rounded">Retake Image</span>
                  </div>
                </div>
                <div className="p-3">
                  <div className="flex justify-between items-start mb-1">
                    <h3 className="text-sm font-medium text-slate-200">Image 2</h3>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-green-900/30 text-green-400 border border-green-800">Captured</span>
                  </div>
                  <p className="text-xs text-slate-400">Back Panel</p>
                </div>
              </div>

              {/* Card 3 */}
              <div className="bg-slate-900 border border-slate-700 rounded-lg overflow-hidden group">
                <div className="h-32 bg-slate-800 flex items-center justify-center relative">
                  <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                  <div className="absolute inset-0 bg-amber-950/50 flex items-center justify-center cursor-pointer border border-amber-500/50">
                    <span className="text-xs font-medium text-amber-100 bg-amber-900/80 px-2 py-1 rounded">Review Image</span>
                  </div>
                </div>
                <div className="p-3">
                  <div className="flex justify-between items-start mb-1">
                    <h3 className="text-sm font-medium text-slate-200">Image 3</h3>
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-amber-900/30 text-amber-400 border border-amber-800">Needs Review</span>
                  </div>
                  <p className="text-xs text-slate-400">Side Panel</p>
                </div>
              </div>

              {/* Uploaded Images */}
              {uploadedImages.map((img) => (
                <div key={img.id} className="bg-slate-900 border border-slate-700 rounded-lg overflow-hidden group">
                  <div className="h-32 bg-slate-800 flex items-center justify-center relative overflow-hidden">
                    <img src={img.url} alt={`Uploaded ${img.number}`} className="object-cover w-full h-full" />
                    <div className="absolute inset-0 bg-slate-950/50 hidden group-hover:flex items-center justify-center transition-all cursor-pointer">
                      <span className="text-xs font-medium text-white bg-black/60 px-2 py-1 rounded">Retake Image</span>
                    </div>
                  </div>
                  <div className="p-3">
                    <div className="flex justify-between items-start mb-1">
                      <h3 className="text-sm font-medium text-slate-200">Image {img.number}</h3>
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-green-900/30 text-green-400 border border-green-800">Captured</span>
                    </div>
                    <div className="flex justify-between items-center mt-1">
                      <p className="text-xs text-slate-400">Product Image</p>
                      <span className="text-[9px] font-medium px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-800">User captured</span>
                    </div>
                  </div>
                </div>
              ))}

              {/* Add New Action */}
              <label className="cursor-pointer h-full min-h-[188px] bg-slate-900/50 border-2 border-dashed border-slate-700 hover:border-amber-500/50 hover:bg-slate-800/50 rounded-lg flex flex-col items-center justify-center transition-colors group">
                <input 
                  type="file" 
                  accept="image/jpeg, image/png, image/webp" 
                  className="hidden" 
                  onChange={handleImageUpload}
                  multiple
                />
                <div className="w-10 h-10 rounded-full bg-slate-800 group-hover:bg-amber-500/20 flex items-center justify-center mb-3 transition-colors">
                  <svg className="w-5 h-5 text-slate-400 group-hover:text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                </div>
                <span className="text-sm font-medium text-slate-300 group-hover:text-amber-500 transition-colors">Add Product Image</span>
              </label>
            </div>
          </div>
        );
      case 2: // Scan
        return (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-slate-100">Scan Product Declarations</h2>
              <p className="text-slate-400 text-sm mt-1">
                Extracting declaration information from the captured package images. Review extracted values before rules validation.
              </p>
            </div>

            {/* AI Warning */}
            <div className="bg-slate-900 border border-slate-700/60 p-3 rounded-lg flex items-start gap-3">
              <svg className="w-5 h-5 text-slate-400 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <p className="text-xs text-slate-400">
                AI/OCR assists with reading declarations. Final compliance assessment is performed by deterministic rules and reviewed by the inspector.
              </p>
            </div>

            {/* OCR Processing Status */}
            <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 flex justify-between items-center">
              <div>
                <h3 className="text-sm font-medium text-slate-300">OCR Processing Pipeline</h3>
                <p className="text-xs text-slate-500 mt-0.5">Ready for OCR processing</p>
              </div>
              <button className="bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 border border-slate-700 font-medium py-1.5 px-4 rounded text-sm transition-colors cursor-not-allowed opacity-80" disabled>
                Run Extraction
              </button>
            </div>

            {/* Source Images */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Source Images</h3>
              <div className="flex gap-3 overflow-x-auto pb-2">
                {[
                  { id: '1', title: 'Front / Declaration Panel' },
                  { id: '2', title: 'Back Panel' },
                  { id: '3', title: 'Side Panel' },
                ].map((img) => (
                  <div key={img.id} className="min-w-[120px] bg-slate-900 border border-slate-700 rounded-lg overflow-hidden shrink-0">
                    <div className="h-16 bg-slate-800 flex items-center justify-center">
                      <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                    </div>
                    <div className="p-2">
                      <div className="text-[10px] font-medium text-slate-300">Image {img.id}</div>
                      <div className="text-[9px] text-slate-500 truncate">{img.title}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Extracted Declarations */}
            <div className="space-y-3 relative">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Extracted Declarations</h3>
                <span className="text-[10px] bg-amber-900/30 text-amber-500 border border-amber-900/50 px-2 py-0.5 rounded">Sample extraction — OCR not connected</span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { field: 'Manufacturer / Packer / Importer', value: 'ABC Foods Pvt. Ltd.', conf: '89%', src: 'Image 2' },
                  { field: 'Generic / Common Name', value: 'Biscuits', conf: '96%', src: 'Image 1' },
                  { field: 'Net Quantity', value: '200 g', conf: '94%', src: 'Image 1' },
                  { field: 'MRP', value: '₹50.00', conf: '98%', src: 'Image 1' },
                  { field: 'Best Before / Use By', value: 'Best Before 6 Months', conf: '91%', src: 'Image 3' },
                ].map((item, idx) => (
                  <div key={idx} className="bg-slate-900 border border-slate-700/80 rounded-lg p-3">
                    <label className="block text-[11px] font-medium text-slate-400 mb-1">{item.field}</label>
                    <div className="text-sm font-semibold text-slate-100 mb-3">{item.value}</div>
                    <div className="flex justify-between items-center text-[10px] text-slate-500 border-t border-slate-800 pt-2">
                      <span className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                        Confidence: {item.conf}
                      </span>
                      <span>Source: {item.src}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Review Action */}
            <div className="pt-2 border-t border-slate-800/50">
              <label className="flex items-center gap-3 cursor-pointer group">
                <div className="w-5 h-5 rounded border border-slate-600 bg-slate-900 flex items-center justify-center group-hover:border-amber-500">
                  <div className="w-3 h-3 bg-amber-500 rounded-sm opacity-0"></div>
                </div>
                <span className="text-sm text-slate-300">I have reviewed the extracted data and verified it against the source images.</span>
              </label>
            </div>
          </div>
        );
      case 3: // Rules Check
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-slate-100">Rules Validation</h2>
            <p className="text-slate-400 text-sm">Validating extracted data against Legal Metrology Packaged Commodities (LMPC) Rules.</p>
            
            <div className="mt-4 space-y-3">
              <div className="bg-slate-900 border border-green-900/30 p-3 rounded-lg flex items-start gap-3">
                <div className="mt-0.5 text-green-500">✓</div>
                <div>
                  <p className="text-sm font-medium text-slate-200">MRP Declaration</p>
                  <p className="text-xs text-slate-400">Found: "MRP ₹150.00 (incl. of all taxes)"</p>
                </div>
              </div>
              
              <div className="bg-slate-900 border border-red-900/30 p-3 rounded-lg flex items-start gap-3">
                <div className="mt-0.5 text-red-500">✗</div>
                <div>
                  <p className="text-sm font-medium text-slate-200">Net Quantity</p>
                  <p className="text-xs text-slate-400">Missing or illegible declaration of net weight.</p>
                </div>
              </div>
              
              <div className="bg-slate-900 border border-yellow-900/30 p-3 rounded-lg flex items-start gap-3">
                <div className="mt-0.5 text-yellow-500">!</div>
                <div>
                  <p className="text-sm font-medium text-slate-200">Manufacturer Address</p>
                  <p className="text-xs text-slate-400">Partial match found, requires manual verification.</p>
                </div>
              </div>
            </div>
          </div>
        );
      case 4: // Officer Review
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-slate-100">Officer Review</h2>
            <p className="text-slate-400 text-sm">Review the findings and add manual observations.</p>
            
            <div className="space-y-4 mt-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Inspector's Remarks</label>
                <textarea 
                  className="w-full h-24 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:ring-2 focus:ring-amber-500" 
                  placeholder="Enter any additional observations, discrepancies, or notes..."
                ></textarea>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Final Assessment</label>
                <div className="flex gap-3">
                  <label className="flex-1 cursor-pointer">
                    <input type="radio" name="assessment" className="peer sr-only" />
                    <div className="text-center py-2 px-3 bg-slate-900 border border-slate-700 rounded-lg peer-checked:bg-green-900/20 peer-checked:border-green-600 peer-checked:text-green-500 text-slate-400 text-sm font-medium transition-colors">
                      Compliant
                    </div>
                  </label>
                  <label className="flex-1 cursor-pointer">
                    <input type="radio" name="assessment" className="peer sr-only" />
                    <div className="text-center py-2 px-3 bg-slate-900 border border-slate-700 rounded-lg peer-checked:bg-red-900/20 peer-checked:border-red-600 peer-checked:text-red-500 text-slate-400 text-sm font-medium transition-colors">
                      Non-Compliant
                    </div>
                  </label>
                </div>
              </div>
            </div>
          </div>
        );
      case 5: // Report
        return (
          <div className="space-y-4 text-center py-8">
            <div className="w-16 h-16 bg-green-500/10 text-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
            </div>
            <h2 className="text-2xl font-bold text-slate-100">Inspection Complete</h2>
            <p className="text-slate-400 text-sm max-w-sm mx-auto">
              The inspection report has been generated and saved locally. Ready to sync with central database.
            </p>
            
            <div className="mt-8 flex flex-col gap-3 max-w-xs mx-auto">
              <button className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium py-2.5 px-4 rounded-lg transition-colors border border-slate-700">
                View Report PDF
              </button>
              <button 
                onClick={() => setCurrentStep(0)}
                className="w-full bg-amber-600 hover:bg-amber-500 text-white font-medium py-2.5 px-4 rounded-lg transition-colors"
              >
                Start New Inspection
              </button>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      <Stepper currentStep={currentStep} />
      
      <main className="flex-1 p-4 md:p-6 overflow-y-auto">
        <div className="max-w-2xl mx-auto bg-slate-950/50 rounded-xl p-4 md:p-6 border border-slate-800/50 min-h-[400px]">
          {renderStepContent()}
        </div>
      </main>
      
      <footer className="border-t border-slate-800 bg-slate-900 p-4 sticky bottom-0">
        <div className="max-w-2xl mx-auto flex justify-between items-center">
          <button
            onClick={prevStep}
            disabled={currentStep === 0 || currentStep === 5}
            className="px-5 py-2.5 rounded-lg font-medium text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-slate-800 text-slate-300 hover:bg-slate-700"
          >
            Back
          </button>
          
          <div className="flex gap-2">
            {currentStep < 5 && (
              <button
                onClick={nextStep}
                className="px-5 py-2.5 rounded-lg font-medium text-sm transition-colors bg-amber-600 text-white hover:bg-amber-500 shadow-lg shadow-amber-900/20"
              >
                {currentStep === 0 ? 'Begin Inspection' : currentStep === 4 ? 'Complete Inspection' : 'Next Step'}
              </button>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}
