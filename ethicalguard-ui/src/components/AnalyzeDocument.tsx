import { useState } from 'react';
import { ScanSearch, AlertTriangle, CheckCircle, AlertCircle, ChevronDown } from 'lucide-react';
import { analyzeDocument, getErrorMessage } from '../services/api';
import type { AnalyzeDocumentResponse, ChunkAnalysis } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreBar from './ScoreBar';

interface Props {
  activeDocument: string;
}

function ChunkCard({ chunk, index }: { chunk: ChunkAnalysis; index: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div className={`rounded-2xl border p-4 space-y-3 ${chunk.flagged ? 'border-rose-200 bg-rose-50' : 'border-slate-200 bg-white'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          {chunk.flagged
            ? <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0" />
            : <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />}
          <span className="text-xs font-semibold text-slate-500">Chunk {index + 1}</span>
          {chunk.flagged && (
            <span className="text-xs bg-rose-100 text-rose-700 px-2 py-0.5 rounded-full font-medium">Flagged</span>
          )}
        </div>
        <button onClick={() => setOpen(v => !v)} className="text-slate-400 hover:text-slate-600">
          <ChevronDown className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Score bars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <ScoreBar label="Toxicity Safety" value={chunk.toxicity_score} />
        <ScoreBar label="Bias Safety"     value={chunk.bias_score} />
        <ScoreBar label="Ethics Score"    value={chunk.ethics_score} />
        <ScoreBar label="Manipulation Penalty" value={chunk.manipulation_penalty} invert />
      </div>

      {/* Chunk text (expandable) */}
      {open && (
        <div className="bg-white border border-slate-200 rounded-xl p-3">
          <p className="text-xs font-semibold text-slate-400 mb-1">Chunk Text</p>
          <p className="text-sm text-slate-600 leading-relaxed">{chunk.chunk}</p>
        </div>
      )}
    </div>
  );
}

export default function AnalyzeDocument({ activeDocument }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeDocumentResponse | null>(null);
  const [error, setError] = useState('');
  const [showAll, setShowAll] = useState(false);

  async function handleAnalyze() {
    if (!activeDocument) {
      setError('Please upload a document first.');
      return;
    }
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

  const displayChunks = result
    ? (showAll ? result.all_chunks : result.unsafe_chunks)
    : [];

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
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
          {/* Summary bar */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-slate-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-slate-700">{result.total_chunks}</p>
              <p className="text-xs text-slate-500">Total Chunks</p>
            </div>
            <div className={`rounded-xl p-3 text-center ${result.flagged_chunks > 0 ? 'bg-rose-50' : 'bg-emerald-50'}`}>
              <p className={`text-2xl font-bold ${result.flagged_chunks > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                {result.flagged_chunks}
              </p>
              <p className={`text-xs ${result.flagged_chunks > 0 ? 'text-rose-500' : 'text-emerald-500'}`}>Flagged</p>
            </div>
            <div className="bg-emerald-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-emerald-600">{result.total_chunks - result.flagged_chunks}</p>
              <p className="text-xs text-emerald-500">Safe</p>
            </div>
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
            {displayChunks.map((chunk, i) => (
              <ChunkCard key={chunk.chunk_index} chunk={chunk} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
