import { useRef, useState } from 'react';
import {
  ShieldCheck, LayoutDashboard, Upload, MessageSquare,
  ScanSearch, Wand2, GitCompare, Menu, X
} from 'lucide-react';

import Hero from './components/Hero';
import RagStatus from './components/RagStatus';
import UploadDocument from './components/UploadDocument';
import AskDocument from './components/AskDocument';
import AnalyzeDocument from './components/AnalyzeDocument';
import RewriteText from './components/RewriteText';
import CompareGeneration from './components/CompareGeneration';

// ── Navigation items ────────────────────────────────────────────────────────
type Section = 'dashboard' | 'upload' | 'ask' | 'analyze' | 'rewrite' | 'compare';

const NAV: { id: Section; label: string; icon: React.ReactNode }[] = [
  { id: 'dashboard', label: 'Dashboard',  icon: <LayoutDashboard className="w-4 h-4" /> },
  { id: 'upload',    label: 'Upload',     icon: <Upload className="w-4 h-4" /> },
  { id: 'ask',       label: 'Ask',        icon: <MessageSquare className="w-4 h-4" /> },
  { id: 'analyze',   label: 'Analyze',    icon: <ScanSearch className="w-4 h-4" /> },
  { id: 'rewrite',   label: 'Rewrite',    icon: <Wand2 className="w-4 h-4" /> },
  { id: 'compare',   label: 'Compare',    icon: <GitCompare className="w-4 h-4" /> },
];

// ── Section wrapper ─────────────────────────────────────────────────────────
function SectionCard({ title, subtitle, children }: {
  title: string; subtitle: string; children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
      <div>
        <h2 className="text-xl font-bold text-slate-800">{title}</h2>
        <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>
      </div>
      {children}
    </div>
  );
}

// ── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [active, setActive] = useState<Section>('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadedDoc, setUploadedDoc] = useState('');
  const mainRef = useRef<HTMLDivElement>(null);

  function navigate(id: Section) {
    setActive(id);
    setSidebarOpen(false);
    mainRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function handleStart() {
    navigate('upload');
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/30 to-violet-50/20 flex">

      {/* ── Sidebar ── */}
      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`
        fixed top-0 left-0 h-full w-60 bg-white border-r border-slate-200 z-30
        flex flex-col shadow-xl transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static lg:shadow-none
      `}>
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-5 py-5 border-b border-slate-100">
          <ShieldCheck className="w-7 h-7 text-indigo-600" strokeWidth={1.5} />
          <div>
            <p className="font-extrabold text-slate-800 leading-none">EthicalGuard</p>
            <p className="text-[10px] text-slate-400 mt-0.5">AI Safety System</p>
          </div>
          <button className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV.map(item => (
            <button
              key={item.id}
              onClick={() => navigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150
                ${active === item.id
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-200'
                  : 'text-slate-600 hover:bg-slate-100'}`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-100">
          <p className="text-[10px] text-slate-400 leading-relaxed">
            Backend: <span className="font-mono">127.0.0.1:8000</span>
          </p>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar (mobile) */}
        <header className="lg:hidden flex items-center gap-3 px-4 py-3 bg-white border-b border-slate-200 sticky top-0 z-10">
          <button onClick={() => setSidebarOpen(true)}>
            <Menu className="w-5 h-5 text-slate-600" />
          </button>
          <ShieldCheck className="w-5 h-5 text-indigo-600" />
          <span className="font-bold text-slate-800">EthicalGuard</span>
        </header>

        <main ref={mainRef} className="flex-1 overflow-y-auto">
          {/* Dashboard = Hero + RAG status */}
          {active === 'dashboard' && (
            <div className="space-y-6 p-6">
              <Hero onStart={handleStart} />
              <div className="max-w-sm">
                <RagStatus />
              </div>
            </div>
          )}

          {active === 'upload' && (
            <div className="p-6 max-w-2xl">
              <SectionCard
                title="Upload Document"
                subtitle="Ingest a .txt or .pdf file into the vector database for analysis and QA."
              >
                <UploadDocument onUploaded={name => { setUploadedDoc(name); navigate('ask'); }} />
              </SectionCard>
            </div>
          )}

          {active === 'ask' && (
            <div className="p-6 max-w-3xl">
              <SectionCard
                title="Ask Your Document"
                subtitle="Ask natural-language questions grounded in the uploaded document using RAG retrieval."
              >
                <AskDocument activeDocument={uploadedDoc} />
              </SectionCard>
            </div>
          )}

          {active === 'analyze' && (
            <div className="p-6 max-w-3xl">
              <SectionCard
                title="Ethical Document Analysis"
                subtitle="Score every chunk of the uploaded document for toxicity, bias, and manipulation."
              >
                <AnalyzeDocument activeDocument={uploadedDoc} />
              </SectionCard>
            </div>
          )}

          {active === 'rewrite' && (
            <div className="p-6 max-w-3xl">
              <SectionCard
                title="Ethical Rewrite"
                subtitle="Paste any text and get a safer, non-manipulative rewritten version with before/after scores."
              >
                <RewriteText />
              </SectionCard>
            </div>
          )}

          {active === 'compare' && (
            <div className="p-6 max-w-3xl">
              <SectionCard
                title="Baseline vs Ethical Comparison"
                subtitle="See how EthicalGuard's reranking pipeline improves raw LLM output quality and safety."
              >
                <CompareGeneration />
              </SectionCard>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
