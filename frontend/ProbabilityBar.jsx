export default function ProbabilityBar({ pAway, pDraw, pHome, homeTeam, awayTeam }) {
  const away = Math.round(pAway * 100);
  const draw = Math.round(pDraw * 100);
  const home = Math.round(pHome * 100);

  const favorite =
    home >= away && home >= draw ? "home" : away >= draw ? "away" : "draw";

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span className={favorite === "away" ? "text-data" : ""}>{awayTeam}</span>
        <span className={favorite === "draw" ? "text-ink" : ""}>Draw</span>
        <span className={favorite === "home" ? "text-signal" : ""}>{homeTeam}</span>
      </div>
      <div className="flex h-9 rounded-md overflow-hidden border border-pitch-700 shadow-inner">
        <div
          className="bg-data-dim flex items-center justify-center text-xs font-mono text-ink transition-[width] duration-500 ease-out"
          style={{ width: `${away}%` }}
        >
          {away >= 10 && `${away}%`}
        </div>
        <div
          className="bg-pitch-700 flex items-center justify-center text-xs font-mono text-ink-dim transition-[width] duration-500 ease-out"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${draw}%`}
        </div>
        <div
          className="bg-signal flex items-center justify-center text-xs font-mono text-pitch-950 font-semibold transition-[width] duration-500 ease-out"
          style={{ width: `${home}%` }}
        >
          {home >= 10 && `${home}%`}
        </div>
      </div>
    </div>
  );
}
