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

## Feature Engineering# MatchMind — Probabilistic Football Score Prediction

MatchMind is a machine learning system for predicting football match outcomes and score distributions using historical Premier League data.
The project focuses on building a **probabilistic modeling pipeline** that goes beyond simple classification to estimate full goal distributions and derived match probabilities.

---

## Overview

Traditional football models predict match outcomes directly (Home/Draw/Away).
This project instead models the **underlying goal process**, allowing richer and more interpretable predictions.

The pipeline:

Data
→ Feature Engineering
→ Ordinal Goal Models (P(G > k))
→ Goal Distributions
→ Scoreline Probability Matrix
→ Match-Level Probabilities

---

## Objectives

* Predict full probability distributions for:

  * Home goals
  * Away goals
* Derive match-level probabilities:

  * Home / Draw / Away
  * Both Teams To Score (BTTS)
  * Over / Under goals
* Build a **leakage-free, temporally consistent pipeline**
* Evaluate models using proper probabilistic metrics

---

## Feature Engineering

The project emphasizes **strictly leakage-free features**, using only past information.

Key techniques:

* Team-level transformation (two rows per match)
* Rolling statistics (3, 5, 15 match windows)
* Rolling shooting efficiency (SoT%)
* Goal difference and form features
* Elo-based team strength ratings
* Matchup features (home attack vs away defense)
* `shift(1)` applied to all rolling features to prevent leakage

---

## Modeling Approach

### Baseline (Outcome Classification)

* XGBoost multiclass classifier
* Time-based train/validation/test split
* Predicts match outcomes directly

#### Baseline Performance

* Accuracy ≈ 50–51%
* LogLoss ≈ 1.10
* Outperforms random and naive baselines

---

### Final Model (Probabilistic Goal Modeling)

Instead of predicting outcomes directly, the model estimates goal distributions using **ordinal classification**.

#### Ordinal Formulation

The model predicts cumulative probabilities:

* P(G > 0)
* P(G > 1)
* P(G > 2)
* P(G > 3)
* P(G > 4)

These are converted into a full distribution over:

0, 1, 2, 3, 4, 5+

Separate models are trained for:

* Home goals
* Away goals

---

## Scoreline Modeling

Goal distributions are combined into a joint scoreline matrix:

P(Home = i, Away = j) = P(Home = i) × P(Away = j)

From this, the model derives:

* Match outcome probabilities (W/D/L)
* BTTS probability
* Over/Under probabilities
* Most likely scoreline
* Expected goals

---

## Results

### Goal Prediction

* Home Goals RPS ≈ 0.123
* Away Goals RPS ≈ 0.123–0.124

### Match-Level Performance

| Metric         | Performance |
| -------------- | ----------- |
| W/D/L Accuracy | ~54–55%     |
| W/D/L Log Loss | ~0.97       |
| BTTS           | weak        |
| Over 2.5       | moderate    |

---

## Key Insight

Modeling goals directly provides a richer and more interpretable framework than direct outcome classification.

However, the assumption:

P(H = i, A = j) = P(H = i) × P(A = j)

introduces limitations, particularly for:

* BTTS
* Over/Under markets

This reflects the fact that scoring in football is **not independent**, but influenced by shared match dynamics such as tempo and game state.

---

## Limitations

* Independence assumption in scoreline modeling
* No explicit modeling of goal correlation
* Limited use of contextual features (e.g., injuries, lineups)

---

## Future Work

* Dixon–Coles style correlation adjustments
* Poisson baseline comparison
* Probabi

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
