# MatchMind — Deployment Guide

## What you have

```
matchmind/
├── backend/            FastAPI app (Render)
│   ├── main.py
│   ├── predict.py
│   ├── build_match_features.py
│   ├── fixtures.py
│   ├── season_simulator.py
│   ├── update_data.py
│   ├── requirements.txt
│   ├── model_a_ensemble/   <- YOU NEED TO ADD THIS (60 .txt files + config.json)
│   └── data/
│       └── top5_leagues_features_full.csv   <- YOU NEED TO ADD THIS
└── frontend/           React + Vite + Tailwind app (Vercel)
```

## 1. Get a football-data.org API key (free)

Sign up at https://www.football-data.org/client/register — free tier gives
10 requests/minute, which is enough for the calendar view and weekly updates.

## 2. Add your model files and data

Copy your `model_a_ensemble/` folder into `backend/model_a_ensemble/`, and
your `top5_leagues_features_full.csv` into `backend/data/`.

## 3. Deploy the backend (Render)

1. Push the `matchmind/` folder to a GitHub repo.
2. On [render.com](https://render.com): New → Web Service → connect the repo,
   root directory `backend/`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Environment variables:
   - `FOOTBALL_DATA_API_KEY` = your key from step 1
   - `MATCHMIND_ADMIN_SECRET` = any password you choose (protects the update endpoint)
   - `MATCHMIND_DATA_PATH` = `data/top5_leagues_features_full.csv`
   - `MATCHMIND_MODEL_DIR` = `model_a_ensemble`
6. **Model files are large (60 files).** If Render's free tier disk/repo
   size becomes an issue, host `model_a_ensemble/` and the CSV on something
   like a private S3 bucket or Git LFS, and download them at container
   startup instead of committing them directly to the repo.
7. Note the deployed URL, e.g. `https://matchmind-api.onrender.com`.

## 4. Deploy the frontend (Vercel)

1. On [vercel.com](https://vercel.com): New Project → same GitHub repo,
   root directory `frontend/`.
2. Framework preset: Vite.
3. Environment variable: `VITE_API_BASE_URL` = your Render backend URL from step 3.
4. Deploy. You'll get a URL like `https://matchmind.vercel.app`.

## 5. Run locally first (recommended before deploying)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY=your_key_here
export MATCHMIND_ADMIN_SECRET=some_password
uvicorn main:app --reload
# -> http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# -> http://localhost:5173 (calls http://localhost:8000 by default)
```

## 6. Weekly data updates

The `/admin/update-data` endpoint only *detects* new results — the real
rebuild (Elo replay + rolling stats) needs `update_data.py` run as an
offline job, since it's too heavy for a single web request.

**Set up a weekly cron job** (Render supports scheduled jobs on paid plans;
alternatively use a free GitHub Action):

```yaml
# .github/workflows/weekly-update.yml
name: Weekly data update
on:
  schedule:
    - cron: "0 6 * * 1"  # every Monday 6am UTC
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt
      - run: python backend/update_data.py --data-path backend/data/top5_leagues_features_full.csv
        env:
          FOOTBALL_DATA_API_KEY: ${{ secrets.FOOTBALL_DATA_API_KEY }}
      - run: |
          git config user.name "matchmind-bot"
          git config user.email "bot@matchmind"
          git add backend/data/top5_leagues_features_full.csv
          git commit -m "Weekly data update" || echo "No changes"
          git push
```

Note: `update_data.py`'s rolling-stat rebuild is scaffolded but not fully
ported from your training notebook (see the comment in that file) — port
your `add_rolling()`/standings logic in before relying on this in production.

## 7. Known gaps to close before this is fully production-ready

- **Team name matching**: football-data.org's team names (e.g. "Manchester
  United FC") may not exactly match your historical CSV's names (e.g.
  "Manchester United"). Build a name-mapping dict in `fixtures.py` if
  `/calendar` predictions start erroring — check the `error` field returned
  per fixture to spot mismatches.
- **Missing match stats for new results**: football-data.org's free tier
  doesn't provide shots/xG/PPDA. Newly appended rows will have NaN there
  until backfilled from your original stats source (likely understat).
- **`season_progress` feature**: still a placeholder (see
  `build_match_features.py`) — fill in if it matters to prediction quality.
- **Validate `build_match_features.py`** against known historical matches
  (see `validate_against_known_match()`) before trusting live predictions.
