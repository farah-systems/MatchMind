const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getLeagues: () => request("/leagues"),
  getCalendar: (league, daysAhead = 14) =>
    request(`/calendar?league=${league}&days_ahead=${daysAhead}`),
  predictMatch: (payload) =>
    request("/predict", { method: "POST", body: JSON.stringify(payload) }),
  simulateSeason: (payload) =>
    request("/simulate-season", { method: "POST", body: JSON.stringify(payload) }),
};
