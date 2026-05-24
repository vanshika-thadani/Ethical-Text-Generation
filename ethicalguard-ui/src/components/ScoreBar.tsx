// A labelled progress bar for a single safety score.
// value is expected in [0, 1]. Colour shifts from red → amber → green.

interface Props {
  label: string;
  value: number;       // 0–1
  invert?: boolean;    // true for penalty scores (lower = better)
}

function barColor(v: number, invert: boolean): string {
  const effective = invert ? 1 - v : v;
  if (effective >= 0.75) return 'bg-emerald-500';
  if (effective >= 0.5)  return 'bg-amber-400';
  return 'bg-rose-500';
}

export default function ScoreBar({ label, value, invert = false }: Props) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  const color = barColor(value, invert);

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-600">
        <span>{label}</span>
        <span className="font-semibold">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
