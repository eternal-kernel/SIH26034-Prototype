import Header from '@/components/Header';
import InspectionWorkflow from '@/components/InspectionWorkflow';

export default function Home() {
  return (
    <>
      <Header />
      <div className="flex-1 flex flex-col min-h-0 bg-slate-950">
        <InspectionWorkflow />
      </div>
    </>
  );
}
