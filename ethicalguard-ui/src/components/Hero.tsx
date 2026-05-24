import { ShieldCheck, ArrowDown } from 'lucide-react';

interface Props {
  onStart: () => void;
}

export default function Hero({ onStart }: Props) {
  return (
    <section className="relative flex flex-col items-center justify-center text-center px-6 py-24 overflow-hidden">
      {/* Soft gradient blobs */}
      <div className="absolute -top-32 -left-32 w-96 h-96 bg-indigo-200 rounded-full opacity-30 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-violet-200 rounded-full opacity-30 blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-3xl space-y-6">
        <div className="flex items-center justify-center gap-3">
          <ShieldCheck className="w-12 h-12 text-indigo-600" strokeWidth={1.5} />
          <h1 className="text-5xl font-extrabold tracking-tight text-slate-800">
            EthicalGuard
          </h1>
        </div>

        <p className="text-lg text-indigo-600 font-medium">
          AI-powered ethical content analysis, document QA, and safe rewriting
        </p>

        <p className="text-slate-500 text-base leading-relaxed max-w-xl mx-auto">
          Upload documents, ask questions, detect unethical content, and generate
          safer rewritten versions using <strong>RAG</strong> and multi-dimensional
          ethical AI scoring — no fine-tuning required.
        </p>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-2 pt-2">
          {['RAG Retrieval', 'Toxicity Detection', 'Bias Scoring', 'Manipulation Detection', 'Safe Rewriting'].map(f => (
            <span key={f} className="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full font-medium">
              {f}
            </span>
          ))}
        </div>

        <button
          onClick={onStart}
          className="mt-4 inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-8 py-3 rounded-xl shadow-lg transition-all duration-200 hover:shadow-indigo-200 hover:shadow-xl"
        >
          Start Analysis
          <ArrowDown className="w-4 h-4" />
        </button>
      </div>
    </section>
  );
}
