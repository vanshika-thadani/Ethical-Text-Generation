import { useState } from 'react';
import { GitCompare, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { compareGeneration, getErrorMessage } from '../services/api';
import type { CompareResponse } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreBar from './ScoreBar';

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

  function GainBadge({ value, label }: { value: number; label: string }) {
    const positive = value >= 0;
    return (
      <div className={`flex items-center gap-1.5 rounded-xl px-3 py-2 ${positive ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`}>
        {positive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
        <div>
          <p className="text-xs font-semibold">{label}</p>
          <p className="text-sm font-bold">{positive ? '+' : ''}{Math.round(value * 100)}%</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-500">
        Compare what a raw LLM produces vs what EthicalGuard's reranking pipeline selects.
      </p>

      <div className="flex gap-3">
        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleCompare()}
          placeholder="Enter a prompt to compare…"
          className="flex-1 border border-slate-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          onClick={handleCompare}
          disabled={loading || !prompt.trim()}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors"
        >
          <GitCompare className="w-4 h-4" />
          Compare
        </button>
      </div>

      {loading && <LoadingSpinner label="Running baseline and ethical reranking…" />}

      {error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}

      {result && !loading && (
        <div className="space-y-5">
          {/* Prompt risk */}
          <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5">
            <span className="text-xs text-slate-500">Prompt risk:</span>
            <span className={`text-sm font-bold ${result.prompt_risk > 0.5 ? 'text-rose-600' : 'text-emerald-600'}`}>
              {Math.round(result.prompt_risk * 100)}%
            </span>
          </div>

          {/* Improvement metrics */}
          <div className="grid grid-cols-3 gap-3">
            <GainBadge value={result.improvement.toxicity_safety_gain} label="Toxicity Safety Gain" />
            <GainBadge value={result.improvement.bias_safety_gain}     label="Bias Safety Gain" />
            <GainBadge value={result.improvement.final_score_gain}     label="Final Score Gain" />
          </div>

          {/* Side-by-side outputs */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Baseline (raw LLM)</p>
              <p className="text-sm text-slate-700 leading-relaxed">{result.baseline_output}</p>
              <div className="space-y-2 pt-2 border-t border-slate-200">
                <ScoreBar label="Toxicity Safety" value={result.baseline_scores.toxicity_score} />
                <ScoreBar label="Ethics Score"    value={result.baseline_scores.ethics_score} />
                <ScoreBar label="Final Score"     value={result.baseline_scores.final_score} />
              </div>
            </div>
            <div className="bg-indigo-50 border border-indigo-200 rounded-2xl p-4 space-y-3">
              <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wide">Safety-Ranked (EthicalGuard)</p>
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
