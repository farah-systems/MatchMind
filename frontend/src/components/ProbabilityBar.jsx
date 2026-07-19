export default function ProbabilityBar({ pAway, pDraw, pHome, homeTeam, awayTeam }) {
  const away = Math.round(pAway * 100);
  const draw = Math.round(pDraw * 100);
  const home = Math.round(pHome * 100);

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span>{awayTeam}</span>
        <span>Draw</span>
        <span>{homeTeam}</span>
      </div>
      <div className="flex h-8 rounded-sm overflow-hidden border border-pitch-700">
        <div
          className="bg-data-dim flex items-center justify-center text-xs font-mono text-ink transition-all"
          style={{ width: `${away}%` }}
        >
          {away >= 10 && `${away}%`}
        </div>
        <div
          className="bg-pitch-700 flex items-center justify-center text-xs font-mono text-ink-dim transition-all"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${draw}%`}
        </div>
        <div
          className="bg-signal flex items-center justify-center text-xs font-mono text-pitch-950 font-medium transition-all"
          style={{ width: `${home}%` }}
        >
          {home >= 10 && `${home}%`}
        </div>
      </div>
    </div>
  );
}
