import { useEffect, useState } from "react";

// Animates a number counting up from 0 on mount/change — gives the
// prediction a sense of the model "arriving" at its answer.
function useCountUp(target, duration = 700) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(target * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

export default function ProbabilityBar({ pAway, pDraw, pHome, homeTeam, awayTeam }) {
  const away = Math.round(pAway * 100);
  const draw = Math.round(pDraw * 100);
  const home = Math.round(pHome * 100);

  const awayAnim = useCountUp(away);
  const drawAnim = useCountUp(draw);
  const homeAnim = useCountUp(home);

  const favorite = home >= away && home >= draw ? "home" : away >= draw ? "away" : "draw";

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span className={favorite === "away" ? "text-pulse-bright" : ""}>{awayTeam}</span>
        <span className={favorite === "draw" ? "text-ink" : ""}>Draw</span>
        <span className={favorite === "home" ? "text-floodlight" : ""}>{homeTeam}</span>
      </div>
      <div className="flex h-9 rounded-md overflow-hidden border border-night-700">
        <div
          className="relative bg-pulse-dim flex items-center justify-center text-xs font-mono text-ink transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${away}%` }}
        >
          {favorite === "away" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{away >= 10 && `${awayAnim}%`}</span>
        </div>
        <div
          className="bg-night-700 flex items-center justify-center text-xs font-mono text-ink-dim transition-[width] duration-700 ease-out"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${drawAnim}%`}
        </div>
        <div
          className="relative bg-floodlight flex items-center justify-center text-xs font-mono text-night-950 font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${home}%` }}
        >
          {favorite === "home" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{home >= 10 && `${homeAnim}%`}</span>
        </div>
      </div>
    </div>
  );
}
