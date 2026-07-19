import { useState } from "react";
import Nav from "./components/Nav";
import CalendarView from "./components/CalendarView";
import SimulateMatch from "./components/SimulateMatch";
import SimulateSeason from "./components/SimulateSeason";

export default function App() {
  const [view, setView] = useState("calendar");

  return (
    <div className="min-h-screen">
      <Nav view={view} setView={setView} />
      {view === "calendar" && <CalendarView />}
      {view === "simulate" && <SimulateMatch />}
      {view === "season" && <SimulateSeason />}
    </div>
  );
}
