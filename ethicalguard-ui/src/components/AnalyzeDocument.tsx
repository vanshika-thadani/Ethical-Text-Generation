import { useState } from 'react';
import { ScanSearch, AlertTriangle, CheckCircle, AlertCircle, ChevronDown, Wand2, RefreshCw } from 'lucide-react';
import { analyzeDocument, rewriteText, getErrorMessage } from '../services/api';
import type { AnalyzeDocumentResponse, ChunkAnalysis } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreBar from './ScoreBar';

interface Props {
  activeDocument: string;
}

// ── Severity helpers ────────────────────────────────────────────────────────

const SEVERITY_CONFIG = {
  HIGH:   { bg: 'bg-rose-50',   border: 'border-rose-300',   badge: 'bg-rose-100 text-rose-700',   highlight: 'bg-rose-100',   label: 'HIGH RISK' },
  MEDIUM: { bg: 'bg-orange-50', border: 'border-orange-300', badge: 'bg-orange-100 text-orange-700', highlight: 'bg-orange-100', label: 'MEDIUM RISK' },
  LOW:    { bg: 'bg-white',     border: 'border-slate-200',  badge: 'bg-emerald-100 text-emerald-700', highlight: '', label: 'SAFE' },
};

// ── Chunk card ──────────────────────────────────────────────────────────────

function ChunkCard({ chunk }: { chunk: ChunkAnalysis }) {
  const [open, setOpen] = useState(false);
  const [rewriting, setRewriting] = useState(false);
  const [rewrite, setRewrite] = useState('');
  const [rewriteError, setRewriteError] = useState('');

  const cfg = SEVERITY_CONFIG[chunk.severity] ?? SEVERITY_CONFIG.LOW;

  // Console log for debugging — visible in browser DevTools
  console.log(
    `[EthicalGuard] Chunk ${chunk.chunk_index} | ` +
    `toxicity_risk=${chunk.toxicity_risk?.toFixed(3)} ` +
    `bias_risk=${(1 - chunk.bias_score).toFixed(3)} ` +
    `manipulation_penalty=${chunk.manipulation_penalty?.toFixed(3)} ` +
    `severity=${chunk.severity}`
  );

  async function handleInlineRewrite() {
    setRewriting(true);
    setRewriteError('');
    setRewrite('');
    try {
      const res = await rewriteText(chunk.chunk);
      setRewrite(res.ethical_rewrite);
    } catch (e) {
      setRewriteError(getErrorMessage(e));
    } finally {
      setRewriting(false);
    }
  }

  return (
    <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} p-4 space-y-3`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          {chunk.flagged
            ? <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0" />
            : <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />}
          <span className="text-xs font-semibold text-slate-500">Chunk {chunk.chunk_index + 1}</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${cfg.badge}`}>
            {cfg.label}
          </span>
          {/* Colour-coded risk type badges */}
          {chunk.toxicity_score < 0.6 && (
            <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full font-medium">Toxic</span>
          )}
          {chunk.manipulation_penalty > 0.1 && (
            <span className="text-xs bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full font-medium">Manipulative</span>
          )}
          {chunk.bias_score < 0.6 && (
            <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full font-medium">Biased</span>
          )}
        </div>
        <button onClick={() => setOpen(v => !v)} className="text-slate-400 hover:text-slate-600 shrink-0">
          <ChevronDown className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Score bars — Risk bars use invert=true so red = high risk */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <ScoreBar label="Toxicity Risk"        value={chunk.toxicity_risk ?? (1 - chunk.toxicity_score)} invert />
        <ScoreBar label="Bias Risk"            value={1 - chunk.bias_score} invert />
        <ScoreBar label="Ethics Score"         value={chunk.ethics_score} />
        <ScoreBar label="Manipulation Penalty" value={chunk.manipulation_penalty} invert />
        <ScoreBar label="Manipulation Penalty" value={chunk.manipulation_penalty} invert />
      </div>

      {/* Expandable chunk text with colour highlight */}
      {open && (
        <div className={`rounded-xl border border-slate-200 p-3 ${cfg.highlight}`}>
          <p className="text-xs font-semibold text-slate-400 mb-1">Chunk Text</p>
          <p className="text-sm text-slate-700 leading-relaxed">{chunk.chunk}</p>
        </div>
      )}

      {/* Inline rewrite button — only for flagged chunks */}
      {chunk.flagged && (
        <div className="pt-1 space-y-2">
          <button
            onClick={handleInlineRewrite}
            disabled={rewriting}
            className="flex items-center gap-1.5 text-xs bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-3 py-1.5 rounded-lg transition-colors"
          >
            {rewriting
              ? <RefreshCw className="w-3 h-3 animate-spin" />
              : <Wand2 className="w-3 h-3" />}
            {rewriting ? 'Rewriting…' : 'Rewrite Safely'}
          </button>

          {rewriteError && (
            <p className="text-xs text-rose-600">{rewriteError}</p>
          )}

          {rewrite && !rewriting && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3">
              <p className="text-xs font-semibold text-emerald-600 mb-1">Safe Rewrite</p>
              <p className="text-sm text-slate-700 leading-relaxed">{rewrite}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function AnalyzeDocument({ activeDocument }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeDocumentResponse | null>(null);
  const [error, setError] = useState('');
  const [showAll, setShowAll] = useState(false);

  async function handleAnalyze() {
    if (!activeDocument) { setError('Please upload a document first.'); return; }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setResult(await analyzeDocument(activeDocument));
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  const displayChunks = result ? (showAll ? result.all_chunks : result.unsafe_chunks) : [];
  const highCount   = result?.all_chunks.filter(c => c.severity === 'HIGH').length ?? 0;
  const mediumCount = result?.all_chunks.filter(c => c.severity === 'MEDIUM').length ?? 0;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          {activeDocument
            ? <p className="text-sm text-slate-500">Document: <span className="font-medium text-slate-700">{activeDocument}</span></p>
            : <p className="text-sm text-slate-400">Upload a document first to analyse it.</p>}
        </div>
        <button
          onClick={handleAnalyze}
          disabled={loading || !activeDocument}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors"
        >
          <ScanSearch className="w-4 h-4" />
          Analyse Document
        </button>
      </div>

      {loading && <LoadingSpinner label="Scoring every chunk for toxicity, bias, and manipulation…" />}

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-4">
          {/* Summary grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-slate-700">{result.total_chunks}</p>
              <p className="text-xs text-slate-500">Total</p>
            </div>
            <div className="bg-rose-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-rose-600">{highCount}</p>
              <p className="text-xs text-rose-500">High Risk</p>
            </div>
            <div className="bg-orange-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-orange-600">{mediumCount}</p>
              <p className="text-xs text-orange-500">Medium Risk</p>
            </div>
            <div className="bg-emerald-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-emerald-600">{result.total_chunks - result.flagged_chunks}</p>
              <p className="text-xs text-emerald-500">Safe</p>
            </div>
          </div>

          {/* Colour legend */}
          <div className="flex flex-wrap gap-3 text-xs">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-red-400 inline-block" />Red = Toxic</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-orange-400 inline-block" />Orange = Manipulative</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-yellow-400 inline-block" />Yellow = Biased</span>
          </div>

          {/* Toggle */}
          <div className="flex gap-2">
            <button
              onClick={() => setShowAll(false)}
              className={`text-sm px-4 py-1.5 rounded-full font-medium transition-colors ${!showAll ? 'bg-rose-100 text-rose-700' : 'text-slate-500 hover:bg-slate-100'}`}
            >
              Flagged ({result.flagged_chunks})
            </button>
            <button
              onClick={() => setShowAll(true)}
              className={`text-sm px-4 py-1.5 rounded-full font-medium transition-colors ${showAll ? 'bg-indigo-100 text-indigo-700' : 'text-slate-500 hover:bg-slate-100'}`}
            >
              All ({result.total_chunks})
            </button>
          </div>

          {displayChunks.length === 0 && (
            <div className="flex items-center gap-2 text-emerald-600 bg-emerald-50 rounded-2xl p-4">
              <CheckCircle className="w-5 h-5" />
              <p className="text-sm font-medium">No unsafe chunks detected. Document looks clean!</p>
            </div>
          )}

          <div className="space-y-3">
            {displayChunks.map(chunk => (
              <ChunkCard key={chunk.chunk_index} chunk={chunk} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
