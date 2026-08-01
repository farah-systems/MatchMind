import { useEffect, useState } from "react";
import { Swords, CalendarIcon, ShieldCheck, AlertCircle } from "lucide-react";
import { api } from "../api";
import ProbabilityBar from "./ProbabilityBar";

const LEAGUE_NAMES = {
  epl: "Premier League",
  spa: "LaLiga",
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
    <div className="flex items-center gap-2 mb-2.5">
      <span className="w-5 h-5 flex items-center justify-center bg-void-raised border border-line text-[10px] font-mono text-ink-dim">
        {n}
      </span>
      <p className="text-[11px] uppercase tracking-[0.14em] text-ink-dim font-mono">{text}</p>
    </div>
  );
}

function ChoiceGrid({ options, value, onChange, columns = 3 }) {
  // options can be plain strings (team names, where value === label) or
  // {value, label} objects (leagues, where the code and display name differ)
  const normalized = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));
  return (
    <div className={`grid grid-cols-${columns} gap-2`}>
      {normalized.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`px-3 py-2.5 text-sm text-left border transition-colors duration-150 font-body ${
            value === opt.value
              ? "border-magma-hot text-magma-hot-bright bg-magma-hot/10"
              : "border-line text-ink-dim hover:text-ink hover:border-line-bright"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// Once a team is chosen, collapse the grid down to just that team's pill
// (less clutter once you've decided) -- clicking the pill again reopens
// the full grid so you can change your mind.
function CollapsibleChoice({ options, value, onChange, columns = 3 }) {
  if (value) {
    return (
      <button
        onClick={() => onChange(null)}
        className="px-3 py-2.5 text-sm border border-magma-hot text-magma-hot-bright bg-magma-hot/10 text-left w-full sm:w-auto transition-colors duration-150"
      >
        {value} <span className="text-ink-dim text-xs ml-1 font-mono">(change)</span>
      </button>
    );
  }
  return <ChoiceGrid options={options} value={value} onChange={onChange} columns={columns} />;
}

export default function SimulateMatch() {
  const [leagues, setLeagues] = useState([]);
  const [league, setLeague] = useState(null);
  const [matchDate, setMatchDate] = useState("");
  const [homeTeam, setHomeTeam] = useState(null);
  const [awayTeam, setAwayTeam] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [error, setError] = useState(null);

  const LOADING_MESSAGES = [
    "Pulling recent form for both teams…",
    "Weighing head-to-head history…",
    "Factoring in Elo and rest days…",
    "Running the ensemble…",
  ];

  useEffect(() => {
    api.getLeagues().then(setLeagues).catch((e) => setError(e.message));
  }, []);

  // Rotate through status messages while the request is in flight, so a
  // few-second wait doesn't feel like nothing is happening.
  useEffect(() => {
    if (!loading) return;
    let i = 0;
    setLoadingMessage(LOADING_MESSAGES[0]);
    const interval = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      setLoadingMessage(LOADING_MESSAGES[i]);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

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
      <div className="relative data-grid pt-16 pb-10 px-6">
        <div className="max-w-2xl mx-auto relative">
          <div className="flex items-center gap-2 mb-4">
            <Swords size={15} className="text-magma-hot" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-dim font-mono">
              Head-to-head simulator
            </span>
          </div>
          <h1 className="font-display font-800 uppercase text-5xl leading-[0.95] tracking-tight mb-4">
            Any two teams.<br />Any matchday.
          </h1>
          <p className="text-ink-dim text-sm max-w-md font-body">
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
              options={leagues.map((l) => ({ value: l.code, label: l.name }))}
              value={league}
              onChange={(v) => {
                setLeague(v);
                resetBelow(1);
              }}
              columns={2}
            />
            {league && <p className="mt-2 text-xs text-ink-dim font-mono">→ {LEAGUE_NAMES[league]}</p>}
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
                  className="bg-void-panel border border-line pl-9 pr-3 py-2 text-sm text-ink focus:border-magma-hot outline-none font-mono"
                />
              </div>
            </div>
          )}

          {league && matchDate && (
            <div className="stagger-item">
              <StepLabel n={3} text="Home team" />
              <CollapsibleChoice
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
              <CollapsibleChoice
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
              className="cut-corner flex items-center gap-2 bg-magma-hot text-void font-semibold px-5 py-2.5 text-sm hover:bg-magma-hot-bright transition-colors duration-150 disabled:opacity-50 font-mono uppercase tracking-wide"
            >
              <ShieldCheck size={16} />
              {loading ? "Simulating…" : "Simulate match"}
            </button>
          )}

          {loading && (
            <div className="cut-corner-lg bg-void-panel border border-line p-5">
              <div className="h-2 bg-void-raised overflow-hidden mb-3 relative">
                <div className="absolute inset-0 w-1/3 bg-gradient-to-r from-magma-mid to-magma-hot animate-[loadingSlide_1.2s_ease-in-out_infinite]" />
              </div>
              <p className="text-xs font-mono text-ink-dim">{loadingMessage}</p>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 text-sm text-risk bg-risk-dim/20 border border-risk/30 px-4 py-3">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {result && (
            <div
              className="cut-corner-lg bg-void-panel border border-line p-6 mt-4"
              style={{ animation: "fadeIn 0.4s ease-out" }}
            >
              <div className="flex items-center justify-center gap-6 mb-6">
                <div className="flex flex-col items-center gap-2">
                  <div className="cut-corner w-14 h-14 bg-void-raised border border-magma-hot/40 flex items-center justify-center font-display font-700 text-sm">
                    {initials(result.home_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate font-mono">
                    {result.home_team}
                  </span>
                </div>
                <span className="font-display font-700 text-ink-dim text-lg uppercase">vs</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="cut-corner w-14 h-14 bg-void-raised border border-magma-mid/40 flex items-center justify-center font-display font-700 text-sm">
                    {initials(result.away_team)}
                  </div>
                  <span className="text-xs text-ink-dim max-w-[80px] text-center truncate font-mono">
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
