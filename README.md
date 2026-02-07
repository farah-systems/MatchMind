# MatchMind
MatchMind - a machine learning system to predict EPL match outcomes and scores using historical match statistics, feature engineering, and model evaluation.

## Project Overview
This project builds a leakage-free machine learning model to predict Premier League match outcomes and scores using historical data.
The project emphasizes:
- Leakage-free feature engineering
- Temporal validation (no random splits)
- Robust evaluation using probabilistic metrics

## Objectives
- Predict match outcomes (Home/Draw/Away)
- Predict home & away goals
- Analyse feature importance and model confidence
- Transition from classification to score-based modeling

## Feature Engineering
- Team-level transformation (2 rows per match)
- Rolling statistics (3, 5, 15 match windows)
- Rolling SoT% (sum-based computation)
- GamesPlayed feature for early-season stability
- Strict shift(1) to prevent data leakage
All features are computed using only information available before each match.

## Modeling Approach
### Version 1 — Baseline
- XGBoost multiclass classifier
- Time-based train/validation/test split
- No additional interaction features
- No feature scaling (tree-based model)
#### Baseline Results
Validation:
- Accuracy ≈ 50.5%
- LogLoss ≈ 1.106
- Outperforms random baseline (33.3%) by (+50%)
- Outperforms AlwaysHome baseline (42.2%) by (+20%)
- Underperformes BetOddsBased baseline (62.3%) by (-19%)

#### Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib

#### Roadmap
Next versions will introduce:
- Home-away difference features
- Rolling goal difference
- Rest-day features
- Elo ratings
- possession
- Poisson score modeling
- Calibration analysis
