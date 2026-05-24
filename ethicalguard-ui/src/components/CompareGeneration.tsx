import { useState } from 'react';
import { GitCompare, AlertCircle, RefreshCw, ArrowRight } from 'lucide-react';
import { compareGeneration, getErrorMessage } from '../services/api';
import type { CompareResponse } from '../types/api';
import ScoreBar from './ScoreBar';

// Demo examples covering the key use-cases
const DEMO_EXAMPLES = [
  { label: 'Manipulation',         text: 'You must follow my advice or you will regret it.' },
  { label: 'Toxic communication',  text: 'People like them are always causing problems.' },
  { label: 'Biased language',      text: 'Those people are always dishonest and unreliable.' },
  { label: 'Unethical persuasion', text: 'This is the only way. Trust me blindly.' },
];

export default function CompareGeneration() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState('');

  async function handleCompare() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setResult(await compareGeneration(prompt));
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  function MetricRow({ label, before, after }: { label: string; before: number; after: number }) {
    const gain = after - before;
    const positive = gain >= 0;
    return (
      <div className="flex items-center gap-3 text-sm">
        <span className="w-32 text-xs text-slate-500 shrink-0">{label}</span>
        <span className="font-mono text-slate-600 w-10 text-right">{Math.round(before * 100)}%</span>
        <ArrowRight className="w-3 h-3 text-slate-400 shrink-0" />
        <span className={`font-mono font-bold w-10 text-right ${positive ? 'text-emerald-600' : 'text-rose-600'}`}>
          {Math.round(after * 100)}%
        </span>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${positive ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
          {positive ? '+' : ''}{Math.round(gain * 100)}%
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-500">
        See how EthicalGuard's reranking pipeline improves raw LLM output — same prompt, safer result.
      </p>

      {/* Demo examples */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Demo examples</p>
        <div className="flex flex-wrap gap-2">
          {DEMO_EXAMPLES.map(ex => (
            <button
              key={ex.label}
              onClick={() => { setPrompt(ex.text); setResult(null); setError(''); }}
              className="text-xs bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 text-slate-600 hover:text-indigo-700 px-3 py-1.5 rounded-full transition-colors font-medium"
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <input
          value={prompt}
          onChange={e => { setPrompt(e.target.value); setResult(null); }}
          onKeyDown={e => e.key === 'Enter' && handleCompare()}
          placeholder="Enter a prompt to compare baseline vs ethical output…"
          className="flex-1 border border-slate-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          onClick={handleCompare}
          disabled={loading || !prompt.trim()}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors shrink-0"
        >
          {loading
            ? <RefreshCw className="w-4 h-4 animate-spin" />
            : <GitCompare className="w-4 h-4" />}
          {loading ? 'Running…' : 'Compare'}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-5">
          {/* Prompt risk badge */}
          <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5">
            <span className="text-xs text-slate-500">Prompt risk score:</span>
            <span className={`text-sm font-bold px-2 py-0.5 rounded-full ${
              result.prompt_risk > 0.6 ? 'bg-rose-100 text-rose-700'
              : result.prompt_risk > 0.3 ? 'bg-amber-100 text-amber-700'
              : 'bg-emerald-100 text-emerald-700'
            }`}>
              {Math.round(result.prompt_risk * 100)}%
            </span>
            <span className="text-xs text-slate-400">
              {result.prompt_risk > 0.6 ? '⚠ High risk prompt' : result.prompt_risk > 0.3 ? 'Moderate risk' : 'Low risk'}
            </span>
          </div>

          {/* Improvement summary */}
          <div className="bg-white border border-slate-200 rounded-2xl p-4 space-y-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Score Improvements</p>
            <MetricRow
              label="Toxicity Safety"
              before={result.baseline_scores.toxicity_score}
              after={result.safety_ranked_scores.toxicity_score}
            />
            <MetricRow
              label="Bias Safety"
              before={result.baseline_scores.bias_score}
              after={result.safety_ranked_scores.bias_score}
            />
            <MetricRow
              label="Ethics Score"
              before={result.baseline_scores.ethics_score}
              after={result.safety_ranked_scores.ethics_score}
            />
            <MetricRow
              label="Final Score"
              before={result.baseline_scores.final_score}
              after={result.safety_ranked_scores.final_score}
            />
          </div>

          {/* Side-by-side outputs */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Baseline */}
            <div className="bg-slate-50 border border-slate-300 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold bg-slate-200 text-slate-600 px-2 py-0.5 rounded-full">BASELINE</span>
                <span className="text-xs text-slate-400">Raw LLM output</span>
              </div>
              <p className="text-sm text-slate-700 leading-relaxed">{result.baseline_output}</p>
              <div className="space-y-2 pt-2 border-t border-slate-200">
                <ScoreBar label="Toxicity Safety" value={result.baseline_scores.toxicity_score} />
                <ScoreBar label="Ethics Score"    value={result.baseline_scores.ethics_score} />
                <ScoreBar label="Final Score"     value={result.baseline_scores.final_score} />
              </div>
            </div>

            {/* Safety-ranked */}
            <div className="bg-indigo-50 border border-indigo-300 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold bg-indigo-200 text-indigo-700 px-2 py-0.5 rounded-full">ETHICALGUARD</span>
                <span className="text-xs text-indigo-400">Safety-ranked output</span>
              </div>
              <p className="text-sm text-slate-700 leading-relaxed">{result.safety_ranked_output}</p>
              <div className="space-y-2 pt-2 border-t border-indigo-200">
                <ScoreBar label="Toxicity Safety" value={result.safety_ranked_scores.toxicity_score} />
                <ScoreBar label="Ethics Score"    value={result.safety_ranked_scores.ethics_score} />
                <ScoreBar label="Final Score"     value={result.safety_ranked_scores.final_score} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
