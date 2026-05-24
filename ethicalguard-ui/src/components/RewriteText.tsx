import { useState } from 'react';
import { Wand2, AlertCircle, ArrowRight, TrendingUp } from 'lucide-react';
import { rewriteText, getErrorMessage } from '../services/api';
import type { RewriteResponse } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreBar from './ScoreBar';

const EXAMPLES = [
  'You must do this or you will regret it. Everyone knows there is no choice.',
  'People like them are always causing problems. Trust me blindly on this.',
  'This is the only way. Guaranteed results. Don\'t tell anyone about this secret method.',
];

export default function RewriteText() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RewriteResponse | null>(null);
  const [error, setError] = useState('');

  async function handleRewrite() {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setResult(await rewriteText(text));
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  const gain = result
    ? result.scores_after.final_score - result.scores_before.final_score
    : 0;

  return (
    <div className="space-y-5">
      {/* Example buttons */}
      <div className="space-y-1">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Try an example</p>
        <div className="flex flex-col gap-2">
          {EXAMPLES.map(ex => (
            <button
              key={ex}
              onClick={() => setText(ex)}
              className="text-left text-xs bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 text-slate-600 hover:text-indigo-700 px-3 py-2 rounded-xl transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      {/* Textarea */}
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        rows={4}
        placeholder="Paste text to rewrite ethically…"
        className="w-full border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none"
      />

      <button
        onClick={handleRewrite}
        disabled={loading || !text.trim()}
        className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-xl transition-colors"
      >
        <Wand2 className="w-4 h-4" />
        Rewrite Ethically
      </button>

      {loading && <LoadingSpinner label="Generating ethical rewrite…" />}

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-4">
          {/* Score gain banner */}
          <div className={`flex items-center gap-3 rounded-2xl p-4 ${gain >= 0 ? 'bg-emerald-50 border border-emerald-200' : 'bg-amber-50 border border-amber-200'}`}>
            <TrendingUp className={`w-5 h-5 ${gain >= 0 ? 'text-emerald-600' : 'text-amber-600'}`} />
            <div>
              <p className={`font-semibold text-sm ${gain >= 0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                Final score {gain >= 0 ? 'improved' : 'changed'} by {gain >= 0 ? '+' : ''}{Math.round(gain * 100)}%
              </p>
              <p className="text-xs text-slate-500">
                {Math.round(result.scores_before.final_score * 100)}% → {Math.round(result.scores_after.final_score * 100)}%
              </p>
            </div>
          </div>

          {/* Side-by-side text */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 space-y-2">
              <p className="text-xs font-semibold text-rose-500 uppercase tracking-wide">Original</p>
              <p className="text-sm text-slate-700 leading-relaxed">{result.original}</p>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-4 space-y-2">
              <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide">Ethical Rewrite</p>
              <p className="text-sm text-slate-700 leading-relaxed">{result.ethical_rewrite}</p>
            </div>
          </div>

          {/* Score comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Before</p>
              <ScoreBar label="Toxicity Safety"  value={result.scores_before.toxicity_score} />
              <ScoreBar label="Bias Safety"      value={result.scores_before.bias_score} />
              <ScoreBar label="Ethics Score"     value={result.scores_before.ethics_score} />
              <ScoreBar label="Final Score"      value={result.scores_before.final_score} />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide flex items-center gap-1">
                After <ArrowRight className="w-3 h-3" />
              </p>
              <ScoreBar label="Toxicity Safety"  value={result.scores_after.toxicity_score} />
              <ScoreBar label="Bias Safety"      value={result.scores_after.bias_score} />
              <ScoreBar label="Ethics Score"     value={result.scores_after.ethics_score} />
              <ScoreBar label="Final Score"      value={result.scores_after.final_score} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
