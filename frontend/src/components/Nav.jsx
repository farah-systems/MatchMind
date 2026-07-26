import { CalendarDays, Swords, Trophy, Info } from "lucide-react";

export default function Nav({ view, setView }) {
  const tabs = [
    { id: "calendar", label: "Calendar", icon: CalendarDays },
    { id: "simulate", label: "Simulate a match", icon: Swords },
    { id: "season", label: "Simulate a season", icon: Trophy },
    { id: "about", label: "About", icon: Info },
  ];

  return (
    <nav className="border-b border-night-700 sticky top-0 bg-night-950/85 backdrop-blur-md z-20">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="pulse-ring w-2.5 h-2.5 rounded-full bg-floodlight" />
          <div className="font-display font-600 text-xl tracking-tight">
            Match<span className="text-floodlight">Mind</span>
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
                className={`relative flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                  active
                    ? "bg-night-700 text-ink shadow-glow"
                    : "text-ink-dim hover:text-ink hover:bg-night-800"
                }`}
              >
                <Icon size={15} className={active ? "text-pulse-bright" : ""} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
