/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        pitch: {
          950: "#0A100D",
          900: "#0F1712",
          800: "#161F1A",
          700: "#212C25",
        },
        signal: {
          DEFAULT: "#F2994A",
          bright: "#FFB067",
        },
        data: {
          DEFAULT: "#4FD1C5",
          dim: "#2E7D74",
        },
        ink: {
          DEFAULT: "#EDF2EF",
          dim: "#8FA396",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
