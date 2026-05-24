import { useRef, useState } from 'react';
import {
  ShieldCheck, LayoutDashboard, Upload, MessageSquare,
  ScanSearch, Wand2, GitCompare, Menu, X, Image, Info
} from 'lucide-react';

import Hero from './components/Hero';
import RagStatus from './components/RagStatus';
import UploadDocument from './components/UploadDocument';
import AskDocument from './components/AskDocument';
import AnalyzeDocument from './components/AnalyzeDocument';
import RewriteText from './components/RewriteText';
import CompareGeneration from './components/CompareGeneration';

// ── Navigation ──────────────────────────────────────────────────────────────
type Section = 'dashboard' | 'upload' | 'ask' | 'analyze' | 'rewrite' | 'compare' | 'multimodal';

const NAV: { id: Section; label: string; icon: React.ReactNode; badge?: string }[] = [
  { id: 'dashboard',   label: 'Dashboard',   icon: <LayoutDashboard className="w-4 h-4" /> },
  { id: 'upload',      label: 'Upload',      icon: <Upload className="w-4 h-4" /> },
  { id: 'ask',         label: 'Ask',         icon: <MessageSquare className="w-4 h-4" /> },
  { id: 'analyze',     label: 'Analyze',     icon: <ScanSearch className="w-4 h-4" /> },
  { id: 'rewrite',     label: 'Rewrite',     icon: <Wand2 className="w-4 h-4" /> },
  { id: 'compare',     label: 'Compare',     icon: <GitCompare className="w-4 h-4" /> },
  { id: 'multimodal',  label: 'Image Mod.',  icon: <Image className="w-4 h-4" />, badge: 'Soon' },
];

// ── Demo banner ─────────────────────────────────────────────────────────────
function DemoBanner() {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;
  return (
    <div className="flex items-start gap-3 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 mx-6 mt-4">
      <Info className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
      <p className="text-xs text-amber-700 leading-relaxed flex-1">
        <span className="font-semibold">Demo mode:</span> Current demo uses lightweight instruction-tuned local models.
        Rewrite quality may vary while ethical scoring and RAG pipelines remain fully functional.
      </p>
      <button onClick={() => setDismissed(true)} className="text-amber-400 hover:text-amber-600 shrink-0">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

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

// ── Multimodal placeholder ──────────────────────────────────────────────────
function MultimodalPlaceholder() {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-10 text-center space-y-4">
      <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto">
        <Image className="w-8 h-8 text-slate-400" />
      </div>
      <div>
        <h2 className="text-xl font-bold text-slate-700">Image Moderation</h2>
        <p className="text-sm text-slate-400 mt-1">Coming Soon</p>
      </div>
      <p className="text-sm text-slate-500 max-w-md mx-auto leading-relaxed">
        Ethical analysis of images for harmful, biased, or manipulative visual content.
        This module will integrate with the existing RAG and scoring pipeline to provide
        multi-modal content safety analysis.
      </p>
      <div className="flex flex-wrap justify-center gap-2 pt-2">
        {['Harmful content detection', 'Bias in imagery', 'Manipulative visuals', 'Browser extension ready'].map(f => (
          <span key={f} className="text-xs bg-slate-100 text-slate-500 border border-slate-200 px-3 py-1 rounded-full">
            {f}
          </span>
        ))}
      </div>
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/30 to-violet-50/20 flex">

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/30 z-20 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ── Sidebar ── */}
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
            <p className="text-[10px] text-slate-400 mt-0.5">AI Content Safety Platform</p>
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
              <span className="flex-1 text-left">{item.label}</span>
              {item.badge && (
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                  active === item.id ? 'bg-white/20 text-white' : 'bg-amber-100 text-amber-600'
                }`}>
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Sidebar footer */}
        <div className="px-5 py-4 border-t border-slate-100 space-y-1">
          <p className="text-[10px] text-slate-400">Backend: <span className="font-mono">127.0.0.1:8000</span></p>
          <p className="text-[10px] text-slate-300">Browser extension: coming soon</p>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar */}
        <header className="lg:hidden flex items-center gap-3 px-4 py-3 bg-white border-b border-slate-200 sticky top-0 z-10">
          <button onClick={() => setSidebarOpen(true)}>
            <Menu className="w-5 h-5 text-slate-600" />
          </button>
          <ShieldCheck className="w-5 h-5 text-indigo-600" />
          <span className="font-bold text-slate-800">EthicalGuard</span>
        </header>

        <main ref={mainRef} className="flex-1 overflow-y-auto">
          <DemoBanner />

          <div className="p-6">
            {active === 'dashboard' && (
              <div className="space-y-6">
                <Hero onStart={() => navigate('upload')} />
                <div className="max-w-sm">
                  <RagStatus />
                </div>
              </div>
            )}

            {active === 'upload' && (
              <div className="max-w-2xl">
                <SectionCard
                  title="Upload Document"
                  subtitle="Ingest a .txt or .pdf file into the vector database for ethical analysis and RAG-based QA."
                >
                  <UploadDocument onUploaded={name => { setUploadedDoc(name); navigate('ask'); }} />
                </SectionCard>
              </div>
            )}

            {active === 'ask' && (
              <div className="max-w-3xl">
                <SectionCard
                  title="Ask Your Document"
                  subtitle="Retrieve relevant passages using semantic search and generate ethically scored answers."
                >
                  <AskDocument activeDocument={uploadedDoc} />
                </SectionCard>
              </div>
            )}

            {active === 'analyze' && (
              <div className="max-w-3xl">
                <SectionCard
                  title="Ethical Document Analysis"
                  subtitle="Detect toxic, biased, and manipulative content in every chunk. Click 'Rewrite Safely' to fix flagged sections inline."
                >
                  <AnalyzeDocument activeDocument={uploadedDoc} />
                </SectionCard>
              </div>
            )}

            {active === 'rewrite' && (
              <div className="max-w-3xl">
                <SectionCard
                  title="Safe Rewriting"
                  subtitle="Paste any harmful, manipulative, or biased text and get an ethically rewritten version with before/after safety scores."
                >
                  <RewriteText />
                </SectionCard>
              </div>
            )}

            {active === 'compare' && (
              <div className="max-w-3xl">
                <SectionCard
                  title="Baseline vs Ethical Comparison"
                  subtitle="Compare raw LLM output against EthicalGuard's safety-ranked output on the same prompt."
                >
                  <CompareGeneration />
                </SectionCard>
              </div>
            )}

            {active === 'multimodal' && (
              <div className="max-w-2xl">
                <MultimodalPlaceholder />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
