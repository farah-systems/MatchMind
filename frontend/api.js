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
  getCalendar: (league, daysAhead = 60) =>
    request(`/calendar?league=${league}&days_ahead=${daysAhead}`),
  predictMatch: (payload) =>
    request("/predict", { method: "POST", body: JSON.stringify(payload) }),

  // Season simulation is a background job on the backend (so the
  // frontend can show live "N / 5,000 simulations" progress instead of
  // one long blocking request). startSeasonSimulation kicks it off and
  // returns a job_id; poll getSeasonSimulationStatus with that id.
  startSeasonSimulation: (payload) =>
    request("/simulate-season/start", { method: "POST", body: JSON.stringify(payload) }),
  getSeasonSimulationStatus: (jobId) =>
    request(`/simulate-season/status/${jobId}`),

  // Convenience wrapper: starts the job and polls until done/error,
  // calling onProgress(completed, total) along the way.
  simulateSeason: (payload, onProgress) => {
    return api.startSeasonSimulation(payload).then(({ job_id, total }) => {
      onProgress?.(0, total);
      return new Promise((resolve, reject) => {
        const poll = () => {
          api
            .getSeasonSimulationStatus(job_id)
            .then((status) => {
              onProgress?.(status.completed, status.total);
              if (status.status === "done") {
                resolve(status.result);
              } else if (status.status === "error") {
                reject(new Error(status.error || "Simulation failed"));
              } else {
                setTimeout(poll, 400);
              }
            })
            .catch(reject);
        };
        poll();
      });
    });
  },
};
