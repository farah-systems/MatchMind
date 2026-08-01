import { CalendarDays, Swords, Trophy, Info } from "lucide-react";

const TICKER_ITEMS = [
  "5 LEAGUES TRACKED",
  "560+ ENGINEERED FEATURES",
  "LIGHTGBM ENSEMBLE",
  "5,000-TRIAL MONTE CARLO",
  "LIVE FIXTURE FEED",
];

export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar", icon: CalendarDays },
    { id: "simulate", label: "Simulate a match", icon: Swords },
    { id: "season", label: "Simulate a season", icon: Trophy },
    { id: "about", label: "About", icon: Info },
  ];

  const tickerContent = TICKER_ITEMS.join("   ◆   ");

  return (
    <div className="sticky top-0 z-20 bg-void/90 backdrop-blur-md border-b border-line">
      <nav className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="cut-corner w-7 h-7 bg-gradient-to-br from-magma-hot to-magma-mid flex items-center justify-center">
            <span className="font-display font-800 text-void text-sm leading-none">M</span>
          </div>
          <div className="font-display font-700 text-xl tracking-tight uppercase leading-none">
            Match<span className="text-magma-hot">Mind</span>
          </div>
        </div>
        <div className="flex gap-1">
          {tabs.map((t) => {
            const Icon = t.icon;
            const active = view === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setView(t.id)}
                className={`relative flex items-center gap-1.5 px-3.5 py-2 text-xs font-mono uppercase tracking-wide transition-colors duration-150 border-b-2 ${
                  active
                    ? "text-ink border-magma-hot"
                    : "text-ink-dim border-transparent hover:text-ink hover:border-line-bright"
                }`}
              >
                <Icon size={14} className={active ? "text-magma-hot" : ""} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Signature ticker strip -- the one recurring broadcast motif,
          scrolling model vitals under the nav on every view. */}
      <div className="ticker-row overflow-hidden border-t border-line bg-void-panel/60">
        <div className="ticker-track flex whitespace-nowrap py-1.5">
          <span className="font-mono text-[10px] tracking-[0.18em] text-ink-faint px-4">
            {tickerContent}
          </span>
          <span className="font-mono text-[10px] tracking-[0.18em] text-ink-faint px-4">
            {tickerContent}
          </span>
        </div>
      </div>
    </div>
  );
}
