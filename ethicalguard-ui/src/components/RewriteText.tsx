import { useState } from 'react';
import { Wand2, AlertCircle, ArrowRight, TrendingUp, RefreshCw } from 'lucide-react';
import { rewriteText, getErrorMessage } from '../services/api';
import type { RewriteResponse } from '../types/api';
import ScoreBar from './ScoreBar';

// Demo examples covering manipulation, toxicity, bias, and unethical persuasion
const EXAMPLES = [
  { label: 'Manipulation',          text: 'You must follow my advice or you will regret it.' },
  { label: 'Toxic communication',   text: 'People like them are always causing problems. Trust me blindly on this.' },
  { label: 'Biased language',       text: 'Those people from that group are always dishonest and unreliable.' },
  { label: 'Unethical persuasion',  text: 'This is the only way. Guaranteed results. Don\'t tell anyone about this secret method.' },
  { label: 'Emotional aggression',  text: 'You are worthless and everyone knows you will never succeed.' },
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

  const toxGain = result
    ? result.scores_after.toxicity_score - result.scores_before.toxicity_score
    : 0;

  const manipGain = result
    ? result.scores_before.manipulation_penalty - result.scores_after.manipulation_penalty
    : 0;

  return (
    <div className="space-y-5">
      {/* Example buttons */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Try a demo example</p>
        <div className="flex flex-col gap-2">
          {EXAMPLES.map(ex => (
            <button
              key={ex.label}
              onClick={() => { setText(ex.text); setResult(null); setError(''); }}
              className="text-left flex items-start gap-2 bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 px-3 py-2 rounded-xl transition-colors group"
            >
              <span className="text-[10px] font-bold bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded mt-0.5 shrink-0 group-hover:bg-indigo-200">
                {ex.label}
              </span>
              <span className="text-xs text-slate-600 group-hover:text-indigo-700">{ex.text}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Textarea */}
      <textarea
        value={text}
        onChange={e => { setText(e.target.value); setResult(null); }}
        rows={4}
        placeholder="Paste any toxic, manipulative, or biased text here to rewrite it ethically…"
        className="w-full border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none"
      />

      <button
        onClick={handleRewrite}
        disabled={loading || !text.trim()}
        className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-xl transition-colors"
      >
        {loading
          ? <RefreshCw className="w-4 h-4 animate-spin" />
          : <Wand2 className="w-4 h-4" />}
        {loading ? 'Generating safer rewrite…' : 'Rewrite Ethically'}
      </button>

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-4">
          {/* Improvement banner */}
          <div className={`rounded-2xl p-4 border ${gain >= 0 ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className={`w-5 h-5 ${gain >= 0 ? 'text-emerald-600' : 'text-amber-600'}`} />
              <p className={`font-semibold text-sm ${gain >= 0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                Safety score {gain >= 0 ? 'improved' : 'changed'} by {gain >= 0 ? '+' : ''}{Math.round(gain * 100)}%
              </p>
            </div>
            {/* Metric pills */}
            <div className="flex flex-wrap gap-2">
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${toxGain >= 0 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
                Toxicity {toxGain >= 0 ? '↓' : '↑'} {Math.abs(Math.round(toxGain * 100))}%
              </span>
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${manipGain >= 0 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
                Manipulation removed {manipGain >= 0 ? '↓' : '↑'} {Math.abs(Math.round(manipGain * 100))}%
              </span>
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-slate-100 text-slate-600">
                {Math.round(result.scores_before.final_score * 100)}% → {Math.round(result.scores_after.final_score * 100)}%
              </span>
            </div>
          </div>

          {/* Side-by-side text */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 space-y-2">
              <p className="text-xs font-semibold text-rose-500 uppercase tracking-wide">Original</p>
              <p className="text-sm text-slate-700 leading-relaxed">{result.original}</p>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-4 space-y-2">
              <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide flex items-center gap-1">
                Ethical Rewrite <ArrowRight className="w-3 h-3" />
              </p>
              <p className="text-sm text-slate-700 leading-relaxed font-medium">{result.ethical_rewrite}</p>
            </div>
          </div>

          {/* Score comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Before</p>
              <ScoreBar label="Toxicity Safety"      value={result.scores_before.toxicity_score} />
              <ScoreBar label="Bias Safety"          value={result.scores_before.bias_score} />
              <ScoreBar label="Ethics Score"         value={result.scores_before.ethics_score} />
              <ScoreBar label="Manipulation Penalty" value={result.scores_before.manipulation_penalty} invert />
              <ScoreBar label="Final Score"          value={result.scores_before.final_score} />
            </div>
            <div className="bg-white border border-indigo-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-indigo-500 uppercase tracking-wide">After</p>
              <ScoreBar label="Toxicity Safety"      value={result.scores_after.toxicity_score} />
              <ScoreBar label="Bias Safety"          value={result.scores_after.bias_score} />
              <ScoreBar label="Ethics Score"         value={result.scores_after.ethics_score} />
              <ScoreBar label="Manipulation Penalty" value={result.scores_after.manipulation_penalty} invert />
              <ScoreBar label="Final Score"          value={result.scores_after.final_score} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
