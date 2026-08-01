import { useEffect, useState } from "react";
import { Trophy, AlertCircle, Loader2 } from "lucide-react";
import { api } from "../api";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "LaLiga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

function ProgressBar({ completed, total }) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="cut-corner-lg bg-void-panel border border-line p-6">
      <div className="flex items-center gap-2 mb-4 text-sm text-ink-dim font-mono">
        <Loader2 size={16} className="animate-spin text-magma-hot" />
        Running {total.toLocaleString()} Monte Carlo trials…
      </div>
      <div className="h-3 bg-void-raised overflow-hidden mb-2 relative">
        <div
          className="h-full bg-gradient-to-r from-magma-mid to-magma-hot transition-[width] duration-200 ease-out relative overflow-hidden"
          style={{ width: `${pct}%` }}
        >
          <div className="absolute inset-0 shimmer-bar" />
        </div>
      </div>
      <p className="text-xs font-mono text-ink-dim">
        {completed.toLocaleString()} / {total.toLocaleString()} simulations ({pct}%)
      </p>
    </div>
  );
}

function zoneColor(pos, nTeams) {
  if (pos === 1) return "text-magma-hot";
  if (pos <= 4) return "text-magma-hot-bright";
  if (pos >= nTeams - 2) return "text-risk";
  return "text-ink";
}

export default function SimulateSeason() {
  const [league, setLeague] = useState("epl");
  const [season, setSeason] = useState("");
  const [availableSeasons, setAvailableSeasons] = useState([]);
  const [seasonsLoading, setSeasonsLoading] = useState(false);
  const [seasonsError, setSeasonsError] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 5000 });
  const [error, setError] = useState(null);

  // Real, queryable seasons for the selected league -- avoids the user
  // guessing a season string that football-data.org's free tier
  // doesn't actually have data for.
  useEffect(() => {
    setSeasonsLoading(true);
    setSeasonsError(null);
    setSeason("");
    api
      .getSeasons(league)
      .then((res) => {
        setAvailableSeasons(res.seasons);
        if (res.seasons.length) setSeason(res.seasons[res.seasons.length - 1]);
      })
      .catch((e) => setSeasonsError(e.message))
      .finally(() => setSeasonsLoading(false));
  }, [league]);

  const handleSimulate = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress({ completed: 0, total: 5000 });

    api
      .simulateSeason({ league, season }, (completed, total) => setProgress({ completed, total }))
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const nTeams = result?.standings?.length || 0;

  return (
    <div>
      <div className="relative data-grid pt-16 pb-10 px-6">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <Trophy size={15} className="text-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              Monte Carlo · 5,000 trials
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl leading-[0.95] tracking-tight mb-4">
            Play out the rest<br />of the season.
          </h1>
          <p className="text-ink-dim text-sm max-w-lg font-body">
            Every remaining fixture is scored once, then the season is replayed
            thousands of times to see how the table could realistically settle.
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 pb-16">
        <div className="flex gap-3 items-end mb-8 flex-wrap">
          <div>
            <label className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono block mb-1.5">
              League
            </label>
            <select
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="bg-void-panel border border-line px-3 py-2 text-sm focus:border-magma-hot outline-none font-mono"
            >
              {LEAGUES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono block mb-1.5">
              Season
            </label>
            {seasonsLoading ? (
              <div className="bg-void-panel border border-line px-3 py-2 text-sm text-ink-dim w-32 font-mono">
                Loading…
              </div>
            ) : (
              <select
                value={season}
                onChange={(e) => setSeason(e.target.value)}
                disabled={availableSeasons.length === 0}
                className="bg-void-panel border border-line px-3 py-2 text-sm w-32 focus:border-magma-hot outline-none disabled:opacity-50 font-mono"
              >
                {availableSeasons.length === 0 && <option>No data</option>}
                {availableSeasons.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            )}
          </div>
          <button
            onClick={handleSimulate}
            disabled={loading || !season}
            className="cut-corner bg-magma-hot text-void font-semibold px-5 py-2.5 text-sm hover:bg-magma-hot-bright transition-colors duration-150 disabled:opacity-50 font-mono uppercase tracking-wide"
          >
            {loading ? "Simulating…" : "Simulate season"}
          </button>
        </div>

        {seasonsError && (
          <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3 mb-6">
            <AlertCircle size={16} />
            Couldn't load available seasons: {seasonsError}
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3 mb-6">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {loading && <ProgressBar completed={progress.completed} total={progress.total} />}

        {result && !loading && (
          <div
            className="cut-corner-lg bg-void-panel border border-line overflow-hidden"
            style={{ animation: "fadeIn 0.4s ease-out" }}
          >
            <div className="px-4 py-3 border-b border-line text-xs text-ink-dim font-mono flex items-center justify-between flex-wrap gap-2">
              <span>
                {result.matches_played} played · {result.matches_remaining} remaining
              </span>
              <span className="flex items-center gap-3 uppercase tracking-wide text-[10px]">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-magma-hot" /> Title
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-magma-hot-bright" /> Continental
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-risk" /> Relegation
                </span>
              </span>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[11px] uppercase tracking-wide text-ink-dim border-b border-line font-mono">
                  <th className="text-left px-4 py-2 font-normal">#</th>
                  <th className="text-left px-4 py-2 font-normal">Team</th>
                  <th className="text-right px-4 py-2 font-normal">Avg pts</th>
                  <th className="text-right px-4 py-2 font-normal">Title %</th>
                  <th className="text-right px-4 py-2 font-normal">Top 4 %</th>
                  <th className="text-right px-4 py-2 font-normal">Releg. %</th>
                </tr>
              </thead>
              <tbody>
                {result.standings.map((s, i) => (
                  <tr
                    key={s.team}
                    className="border-b border-line/50 last:border-0 hover:bg-void-raised/60 transition-colors stagger-item font-body"
                    style={{ animationDelay: `${i * 25}ms` }}
                  >
                    <td className={`px-4 py-2.5 font-mono ${zoneColor(i + 1, nTeams)}`}>{i + 1}</td>
                    <td className="px-4 py-2.5">{s.team}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{s.avg_points}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-magma-hot">
                      {s.title_pct > 0 ? `${s.title_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-magma-hot-bright">
                      {s.top4_pct > 0 ? `${s.top4_pct}%` : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-risk">
                      {s.relegation_pct > 0 ? `${s.relegation_pct}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
