export default function Header() {
  return (
    <header className="bg-slate-900 border-b border-slate-800 p-4 sticky top-0 z-10 shadow-md">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">
            Field Inspection Assistant
          </h1>
          <p className="text-sm text-slate-400 font-medium">Legal Metrology</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Placeholder for menu or user profile */}
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
            <span className="text-xs text-slate-300 font-bold">LM</span>
          </div>
        </div>
      </div>
    </header>
  );
}
