import type { CandidateScores } from '../types/api';
import ScoreBar from './ScoreBar';

interface Props {
  scores: CandidateScores;
  title?: string;
}

// Badge colour based on final_score
function badge(score: number) {
  if (score >= 0.7) return 'bg-emerald-100 text-emerald-700';
  if (score >= 0.5) return 'bg-amber-100 text-amber-700';
  return 'bg-rose-100 text-rose-700';
}

export default function ScoreCard({ scores, title = 'Ethical Scores' }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-slate-700">{title}</h3>
        <span className={`text-xs font-bold px-2 py-1 rounded-full ${badge(scores.final_score)}`}>
          Final {Math.round(scores.final_score * 100)}%
        </span>
      </div>

      <div className="space-y-3">
        <ScoreBar label="Toxicity Safety"  value={scores.toxicity_score} />
        <ScoreBar label="Sentiment"        value={scores.sentiment_score} />
        <ScoreBar label="Bias Safety"      value={scores.bias_score} />
        <ScoreBar label="Coherence"        value={scores.coherence_score} />
        <ScoreBar label="Ethics (composite)" value={scores.ethics_score} />
        <ScoreBar label="Fluency"          value={scores.fluency_score} />
        <ScoreBar label="Manipulation Penalty" value={scores.manipulation_penalty} invert />
      </div>
    </div>
  );
}
