import { useEffect, useState } from "react";

// Animates a number counting up from 0 on mount/change -- gives the
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
      <div className="flex justify-between text-[11px] font-mono text-ink-dim mb-1.5 uppercase tracking-wide">
        <span className={favorite === "home" ? "text-magma-hot" : ""}>{homeTeam}</span>
        <span className={favorite === "draw" ? "text-ink" : ""}>Draw</span>
        <span className={favorite === "away" ? "text-magma-mid" : ""}>{awayTeam}</span>
      </div>
      {/* Confidence gauge -- a single continuous ramp from home (hot
          amber) through draw (neutral) to away (cold magenta/indigo),
          the same colormap logic used for the model's xG heatmaps. */}
      <div className="flex h-9 overflow-hidden border border-line">
        <div
          className="relative flex items-center justify-center text-xs font-mono text-void font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${home}%`, background: "linear-gradient(90deg, #C98426, #F2A93C)" }}
        >
          {favorite === "home" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{home >= 10 && `${homeAnim}%`}</span>
        </div>
        <div
          className="bg-void-raised flex items-center justify-center text-xs font-mono text-ink-dim transition-[width] duration-700 ease-out border-x border-line"
          style={{ width: `${draw}%` }}
        >
          {draw >= 10 && `${drawAnim}%`}
        </div>
        <div
          className="relative flex items-center justify-center text-xs font-mono text-ink font-semibold transition-[width] duration-700 ease-out overflow-hidden"
          style={{ width: `${away}%`, background: "linear-gradient(90deg, #D6336C, #3C2079)" }}
        >
          {favorite === "away" && <div className="absolute inset-0 shimmer-bar" />}
          <span className="relative z-10">{away >= 10 && `${awayAnim}%`}</span>
        </div>
      </div>
    </div>
  );
}
