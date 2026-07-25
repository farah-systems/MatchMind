/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // "Floodlit pitch at night" — deep navy stadium dark, not pure
        // black, so panels read as lit surfaces rather than voids.
        night: {
          950: "#080B14",
          900: "#0D1220",
          800: "#131A2C",
          700: "#1D2740",
        },
        // Floodlight amber — the football/matchday energy accent.
        floodlight: {
          DEFAULT: "#FFB020",
          bright: "#FFCB66",
          dim: "#8A5D18",
        },
        // Electric violet — the AI / model-confidence accent.
        pulse: {
          DEFAULT: "#8B7CFF",
          bright: "#AEA3FF",
          dim: "#453D8A",
        },
        // Reserved for live/positive signals only (in-play, goals, form).
        pitch: {
          DEFAULT: "#00D68F",
          dim: "#0A6647",
        },
        ink: {
          DEFAULT: "#F3F5FA",
          dim: "#8790A8",
        },
      },
      fontFamily: {
        display: ["'Bricolage Grotesque'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(139, 124, 255, 0.35)",
        "glow-amber": "0 0 24px -4px rgba(255, 176, 32, 0.35)",
      },
    },
  },
  plugins: [],
};
