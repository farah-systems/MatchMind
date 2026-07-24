import { useState } from "react";
import { Trophy, AlertCircle, Loader2 } from "lucide-react";
import { api } from "../api";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "La Liga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

function ProgressBar({ completed, total }) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="bg-pitch-900 border border-pitch-700 rounded-md p-5">
      <div className="flex items-center gap-2 mb-3 text-sm text-ink-dim">
        <Loader2 size={16} className="animate-spin text-signal" />
        Running Monte Carlo simulation…
      </div>
      <div className="h-2.5 bg-pitch-800 rounded-full overflow-hidden mb-2">
        <div
          className="h-full bg-signal transition-[width] duration-200 ease-out rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs font-mono text-ink-dim">
        {completed.toLocaleString()} / {total.toLocaleString()} simulations ({pct}%)
      </p>
    </div>
  );
}

function zoneColor(pos, nTeams) {
  if (pos === 1) return "text-signal";
  if (pos <= 4) return "text-data";
  if (pos >= nTeams - 2) return "text-red-400";
  return "text-ink";
}

export default function SimulateSeason() {
  const [league, setLeague] = useState("epl");
  const [season, setSeason] = useState("26-27");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 5000 });
  const [error, setError] = useState(null);

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
    <div className="max-w-3xl mx-auto px-6 py-10">
      <div className="flex items-center gap-2 mb-1">
        <Trophy size={20} className="text-signal" />
        <h1 className="font-display font-600 text-2xl">Simulate a full season</h1>
      </div>
      <p className="text-ink-dim text-sm mb-2">
        Runs 5,000 Monte Carlo trials over remaining fixtures, using each match's
        predicted probability to sample outcomes and project final standings.
      </p>
      <p className="text-ink-dim text-xs mb-8">
        Note: per-match probabilities are computed once from current form/Elo and
        held fixed across trials — in-simulation form changes aren't modeled.
      </p>

      <div className="flex gap-3 items-end mb-8 flex-wrap">
        <div>
          <label className="text-xs uppercase tracking-wide text-ink-dim block mb-1.5">
            League
          </label>
          <select
            value={league}
            onChange={(e) => setLeague(e.target.value)}
            className="bg-pitch-900 border border-pitch-700 rounded-md px-3 py-2 text-sm focus:border-signal outline-none"
          >
            {LEAGUES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs uppercase tracking-wide text-ink-dim block mb-1.5">
            Season
          </label>
          <input
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            placeholder="26-27"
            className="bg-pitch-900 border border-pitch-700 rounded-md px-3 py-2 text-sm w-24 focus:border-signal outline-none"
          />
        </div>
        <button
          onClick={handleSimulate}
          disabled={loading}
          className="bg-signal text-pitch-950 font-medium px-5 py-2.5 rounded-md text-sm hover:bg-signal-bright transition-colors disabled:opacity-50"
        >
          {loading ? "Simulating…" : "Simulate season"}
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3 mb-6">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {loading && <ProgressBar completed={progress.completed} total={progress.total} />}

      {result && !loading && (
        <div className="bg-pitch-900 border border-pitch-700 rounded-md overflow-hidden animate-[fadeIn_0.3s_ease-out]">
          <div className="px-4 py-3 border-b border-pitch-700 text-xs text-ink-dim font-mono flex items-center justify-between">
            <span>
              {result.matches_played} played · {result.matches_remaining} remaining
            </span>
            <span className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-signal" /> Title
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-data" /> Continental
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-400" /> Relegation
              </span>
            </span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs uppercase text-ink-dim border-b border-pitch-700">
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
                  className="border-b border-pitch-700/50 last:border-0 hover:bg-pitch-800/50 transition-colors"
                >
                  <td className={`px-4 py-2.5 font-mono ${zoneColor(i + 1, nTeams)}`}>{i + 1}</td>
                  <td className="px-4 py-2.5">{s.team}</td>
                  <td className="px-4 py-2.5 text-right font-mono">{s.avg_points}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-signal">
                    {s.title_pct > 0 ? `${s.title_pct}%` : "—"}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-data">
                    {s.top4_pct > 0 ? `${s.top4_pct}%` : "—"}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-red-400">
                    {s.relegation_pct > 0 ? `${s.relegation_pct}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
