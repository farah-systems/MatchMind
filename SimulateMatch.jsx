import { useEffect, useState } from "react";
import { Swords, CalendarIcon, ShieldCheck, AlertCircle } from "lucide-react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUE_NAMES = {
  epl: "Premier League",
  spa: "La Liga",
  ger: "Bundesliga",
  ita: "Serie A",
  fra: "Ligue 1",
};

function initials(name) {
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
}

function StepLabel({ n, text }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="w-5 h-5 flex items-center justify-center rounded-full bg-night-700 text-[11px] font-mono text-ink-dim">
        {n}
      </span>
      <p className="text-xs uppercase tracking-wide text-ink-dim">{text}</p>
    </div>
  );
}

function ChoiceGrid({ options, value, onChange, columns = 3 }) {
  return (
    <div className={`grid grid-cols-${columns} gap-2`}>
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-3 py-2.5 text-sm rounded-md border text-left transition-all duration-150 ${
            value === opt
              ? "border-pulse text-pulse-bright bg-pulse/10 shadow-glow"
              : "border-night-700 text-ink-dim hover:text-ink hover:border-ink-dim"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

export default function SimulateMatch() {
  const [leagues, setLeagues] = useState([]);
  const [league, setLeague] = useState(null);
  const [matchDate, setMatchDate] = useState("");
  const [homeTeam, setHomeTeam] = useState(null);
  const [awayTeam, setAwayTeam] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getLeagues().then(setLeagues).catch((e) => setError(e.message));
  }, []);

  const teams = leagues.find((l) => l.code === league)?.teams || [];

  const resetBelow = (level) => {
    if (level <= 1) setMatchDate("");
    if (level <= 2) setHomeTeam(null);
    if (level <= 3) setAwayTeam(null);
    setResult(null);
  };

  const canSubmit = league && matchDate && homeTeam && awayTeam && homeTeam !== awayTeam;

  const handleSimulate = () => {
    setLoading(true);
    setError(null);
    api
      .predictMatch({ league, home_team: homeTeam, away_team: awayTeam, match_date: matchDate })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <div className="relative pitch-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-3">
            <Swords size={16} className="text-pulse-bright" />
            <span className="text-xs uppercase tracking-widest text-ink-dim font-mono">
              Head-to-head simulator
            </span>
          </div>
          <h1 className="font-display font-700 text-4xl tracking-tight mb-3">
            Any two teams.<br />Any matchday.
          </h1>
          <p className="text-ink-dim text-sm max-w-md">
            Pick a league, a date, then two teams — the model builds fresh features
            and scores the matchup in real time.
          </p>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-6 pb-16">
        <div className="space-y-8">
          <div>
            <StepLabel n={1} text="League" />
            <ChoiceGrid
              options={leagues.map((l) => l.code)}
              value={league}
              onChange={(v) => {
                setLeague(v);
                resetBelow(1);
              }}
              columns={2}
            />
            {league && <p className="mt-2 text-xs text-ink-dim">→ {LEAGUE_NAMES[league]}</p>}
          </div>

          {league && (
            <div className="stagger-item">
              <StepLabel n={2} text="Date" />
              <div className="relative w-fit">
                <CalendarIcon
                  size={15}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-dim pointer-events-none"
                />
                <input
                  type="date"
                  value={matchDate}
                  onChange={(e) => {
                    setMatchDate(e.target.value);
                    resetBelow(2);
                  }}
                  className="bg-night-900 border border-night-700 rounded-md pl-9 pr-3 py-2 text-sm text-ink focus:border-pulse outline-none"
                />
              </div>
            </div>
          )}

          {league && matchDate && (
            <div className="stagger-item">
              <StepLabel n={3} text="Home team" />
              <ChoiceGrid
                options={teams}
                value={homeTeam}
                onChange={(v) => {
                  setHomeTeam(v);
                  resetBelow(3);
                }}
              />
            </div>
          )}

          {league && matchDate && homeTeam && (
            <div className="stagger-item">
              <StepLabel n={4} text="Away team" />
              <ChoiceGrid
                options={teams.filter((t) => t !== homeTeam)}
                value={awayTeam}
                onChange={setAwayTeam}
              />
            </div>
          )}

          {canSubmit && (
            <button
              onClick={handleSimulate}
              disabled={loading}
              className="flex items-center gap-2 bg-floodlight text-night-950 font-medium px-5 py-2.5 rounded-md text-sm hover:bg-floodlight-bright hover:shadow-glow-amber transition-all duration-200 disabled:opacity-50"
            >
              <ShieldCheck size={16} />
              {loading ? "Simulating…" : "Simulate match"}
            </button>
          )}

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-400 bg-red-400/5 border border-red-400/20 rounded-md px-4 py-3">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {result && (
            <div
              className="bg-night-900 border border-night-700 rounded-lg p-6 mt-4"
              style={{ animation: "fadeIn 0.4s ease-out" }}
            >
              <div className="flex items-center justify-center gap-6 mb-6">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-14 h-14 rounded-full bg-night-700 border border-floodlight/40 flex items-center justify-center font-display font-600 text-sm">
                    {initials(result.home_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate">
                    {result.home_team}
                  </span>
                </div>
                <span className="font-display text-ink-dim text-lg">vs</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="w-14 h-14 rounded-full bg-night-700 border border-pulse/40 flex items-center justify-center font-display font-600 text-sm">
                    {initials(result.away_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate">
                    {result.away_team}
                  </span>
                </div>
              </div>
              <p className="text-center text-xs font-mono text-ink-dim mb-5">{result.date}</p>
              <ProbabilityBar
                pAway={result.p_away}
                pDraw={result.p_draw}
                pHome={result.p_home}
                homeTeam={result.home_team}
                awayTeam={result.away_team}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
