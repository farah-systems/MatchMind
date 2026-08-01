/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Graphite-black terminal surface, not pure black -- panels
        // read as backlit glass, not voids.
        void: {
          DEFAULT: "#06070A",
          panel: "#0E1015",
          raised: "#161922",
        },
        line: {
          DEFAULT: "#23262F",
          bright: "#383C48",
        },
        ink: {
          DEFAULT: "#F2F0E9",
          dim: "#84868F",
          faint: "#4A4C56",
        },
        // "Magma" ramp -- the same low-to-high colormap analysts use
        // for xG heatmaps and model-confidence surfaces. This IS the
        // model's voice throughout the product: cold indigo (unlikely)
        // through magenta to hot amber (likely).
        magma: {
          cold: "#3C2079",
          mid: "#D6336C",
          hot: "#F2A93C",
          "hot-bright": "#FFCC70",
        },
        risk: {
          DEFAULT: "#E5484D",
          dim: "#5C2224",
        },
      },
      fontFamily: {
        // Condensed, heavy-shouldered display face -- the register of
        // a broadcast scoreboard or a manager's team-sheet graphic,
        // not a generic startup sans.
        display: ["'Big Shoulders Display'", "sans-serif"],
        body: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
      boxShadow: {
        // Hard 1px edge-lit outline instead of a soft blurred glow --
        // reads as an illuminated panel edge, not a diffuse aura.
        "edge-hot": "inset 0 0 0 1px rgba(242,169,60,0.55), 0 0 0 1px rgba(242,169,60,0.12)",
        "edge-cold": "inset 0 0 0 1px rgba(214,51,108,0.5), 0 0 0 1px rgba(214,51,108,0.1)",
      },
    },
  },
  plugins: [],
};
