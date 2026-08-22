import { useState } from 'react';
import { MessageSquare, ChevronDown, AlertCircle, BookOpen, ShieldAlert, ShieldCheck } from 'lucide-react';
import { askQuestion, getErrorMessage } from '../services/api';
import type { AskResponse } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreCard from './ScoreCard';
import ScoreBar from './ScoreBar';

const EXAMPLE_QUESTIONS = [
  'Which parts of this document may be unethical?',
  'Is this document biased?',
  'Summarize the main issue in this document.',
  'How can this content be made safer?',
  'Are there any manipulative statements here?',
];

interface Props {
  activeDocument: string;
}

// Risk badge helper
function RiskBadge({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const color = pct >= 60 ? 'bg-rose-100 text-rose-700 border-rose-200'
              : pct >= 35 ? 'bg-orange-100 text-orange-700 border-orange-200'
              :              'bg-emerald-100 text-emerald-700 border-emerald-200';
  return (
    <div className={`flex items-center justify-between px-3 py-2 rounded-xl border text-sm ${color}`}>
      <span className="font-medium">{label}</span>
      <span className="font-bold">{pct}%</span>
    </div>
  );
}

export default function AskDocument({ activeDocument }: Props) {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState('');
  const [showChunks, setShowChunks] = useState(false);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await askQuestion(question, topK, activeDocument || undefined);
      setResult(res);
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  // Determine overall document risk level from context scores
  const docRiskLevel = result
    ? (result.context_toxicity_risk >= 0.60 || result.context_bias_risk >= 0.60 || result.context_manipulation >= 0.35)
        ? 'HIGH'
      : (result.context_toxicity_risk >= 0.35 || result.context_bias_risk >= 0.35 || result.context_manipulation >= 0.20)
        ? 'MEDIUM'
      : 'LOW'
    : null;

  return (
    <div className="space-y-5">
      {/* Example questions */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map(q => (
          <button
            key={q}
            onClick={() => setQuestion(q)}
            className="text-xs bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 px-3 py-1.5 rounded-full transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input row */}
      <div className="flex gap-3">
        <div className="relative flex-1">
          <MessageSquare className="absolute left-3 top-3 w-4 h-4 text-slate-400" />
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAsk()}
            placeholder="Ask a question about the uploaded document…"
            className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          />
        </div>
        <select
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          className="border border-slate-300 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {[1, 2, 3, 5].map(k => <option key={k} value={k}>Top {k}</option>)}
        </select>
        <button
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors"
        >
          Ask
        </button>
      </div>

      {activeDocument && (
        <p className="text-xs text-slate-400">Searching in: <span className="font-medium text-slate-600">{activeDocument}</span></p>
      )}

      {loading && <LoadingSpinner label="Retrieving and generating answer…" />}

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-4">
          {/* Answer */}
          <div className="bg-indigo-50 border border-indigo-200 rounded-2xl p-5">
            <p className="text-xs font-semibold text-indigo-500 uppercase tracking-wide mb-2">Answer</p>
            <p className="text-slate-700 leading-relaxed">{result.answer}</p>
          </div>

          {/* ── Document Content Risk ── */}
          <div className={`rounded-2xl border p-5 space-y-3 ${
            docRiskLevel === 'HIGH'   ? 'bg-rose-50 border-rose-200' :
            docRiskLevel === 'MEDIUM' ? 'bg-orange-50 border-orange-200' :
                                        'bg-emerald-50 border-emerald-200'
          }`}>
            <div className="flex items-center gap-2">
              {docRiskLevel === 'LOW'
                ? <ShieldCheck className="w-5 h-5 text-emerald-600" />
                : <ShieldAlert className={`w-5 h-5 ${docRiskLevel === 'HIGH' ? 'text-rose-600' : 'text-orange-500'}`} />
              }
              <p className="text-sm font-bold text-slate-700">
                Document Content Risk
                <span className={`ml-2 text-xs font-bold px-2 py-0.5 rounded-full ${
                  docRiskLevel === 'HIGH'   ? 'bg-rose-100 text-rose-700' :
                  docRiskLevel === 'MEDIUM' ? 'bg-orange-100 text-orange-700' :
                                              'bg-emerald-100 text-emerald-700'
                }`}>{docRiskLevel}</span>
              </p>
            </div>
            <p className="text-xs text-slate-500">
              Scores from the actual retrieved document passages — shows how unsafe the source content is.
            </p>
            <div className="space-y-2">
              <RiskBadge value={result.context_toxicity_risk}  label="Toxicity Risk in Document" />
              <RiskBadge value={result.context_bias_risk}       label="Bias Risk in Document" />
              <RiskBadge value={result.context_manipulation}    label="Manipulation Risk in Document" />
            </div>
          </div>

          {/* ── Answer Quality Scores ── */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 space-y-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-indigo-500" />
              <p className="text-sm font-bold text-slate-700">
                Answer Safety Scores
                <span className="ml-2 text-xs text-slate-400 font-normal">
                  — how safe EthicalGuard's generated answer is
                </span>
              </p>
            </div>
            <div className="space-y-2.5">
              <ScoreBar label="Toxicity Safety"      value={result.ethical_scores.toxicity_score} />
              <ScoreBar label="Sentiment"            value={result.ethical_scores.sentiment_score} />
              <ScoreBar label="Bias Safety"          value={result.ethical_scores.bias_score} />
              <ScoreBar label="Coherence"            value={result.ethical_scores.coherence_score} />
              <ScoreBar label="Ethics (composite)"   value={result.ethical_scores.ethics_score} />
              <ScoreBar label="Manipulation Penalty" value={result.ethical_scores.manipulation_penalty} invert />
            </div>
          </div>

          {/* Retrieved chunks toggle */}
          <button
            onClick={() => setShowChunks(v => !v)}
            className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700 transition-colors"
          >
            <BookOpen className="w-4 h-4" />
            {result.retrieved_chunks.length} retrieved chunk{result.retrieved_chunks.length !== 1 ? 's' : ''}
            <ChevronDown className={`w-4 h-4 transition-transform ${showChunks ? 'rotate-180' : ''}`} />
          </button>

          {showChunks && (
            <div className="space-y-3">
              {result.retrieved_chunks.map((c, i) => (
                <div key={i} className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>{c.document} · chunk {c.chunk_index}</span>
                    <span>distance {c.distance.toFixed(3)}</span>
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed line-clamp-4">{c.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
