import { useState } from 'react';
import { MessageSquare, ChevronDown, AlertCircle, BookOpen } from 'lucide-react';
import { askQuestion, getErrorMessage } from '../services/api';
import type { AskResponse } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import ScoreCard from './ScoreCard';

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

          {/* Ethical scores */}
          <ScoreCard scores={result.ethical_scores} title="Answer Ethical Scores" />
        </div>
      )}
    </div>
  );
}
