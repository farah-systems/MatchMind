import { useState } from "react";
import { api } from "../api";

const LEAGUES = [
  { code: "epl", name: "Premier League" },
  { code: "spa", name: "La Liga" },
  { code: "ger", name: "Bundesliga" },
  { code: "ita", name: "Serie A" },
  { code: "fra", name: "Ligue 1" },
];

export default function SimulateSeason() {
  const [league, setLeague] = useState("epl");
  const [season, setSeason] = useState("26-27");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSimulate = () => {
    setLoading(true);
    setError(null);
    api
      .simulateSeason({ league, season })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      <h1 className="font-display font-600 text-2xl mb-1">Simulate a full season</h1>
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
            className="bg-pitch-900 border border-pitch-700 rounded-sm px-3 py-2 text-sm"
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
            className="bg-pitch-900 border border-pitch-700 rounded-sm px-3 py-2 text-sm w-24"
          />
        </div>
        <button
          onClick={handleSimulate}
          disabled={loading}
          className="bg-signal text-pitch-950 font-medium px-5 py-2.5 rounded-sm text-sm hover:bg-signal-bright transition-colors disabled:opacity-50"
        >
          {loading ? "Simulating 5,000 seasons…" : "Simulate season"}
        </button>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {result && (
        <div className="bg-pitch-900 border border-pitch-700 rounded-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-pitch-700 text-xs text-ink-dim font-mono">
            {result.matches_played} played · {result.matches_remaining} remaining
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
                <tr key={s.team} className="border-b border-pitch-700/50 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-ink-dim">{i + 1}</td>
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
