"""
Model A — Football Match Outcome Predictor
============================================
Loads the saved 60-model ensemble (20 seeds x [current 3-class model,
stage1 draw-vs-not-draw, stage2 home-vs-away]) and blends them into a
final calibrated 3-class probability distribution [P(away), P(draw), P(home)].

Usage:
    from predict import ModelA
    model = ModelA("model_a_ensemble")
    probs = model.predict(X)   # X: DataFrame with the required feature_cols
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb


class ModelA:
    def __init__(self, model_dir="model_a_ensemble"):
        with open(f"{model_dir}/config.json") as f:
            self.config = json.load(f)

        self.feature_cols = self.config["feature_cols"]
        self.categorical_features = self.config["categorical_features"]
        self.w_current = self.config["blend_weight_current"]
        self.w_two_stage = self.config["blend_weight_two_stage"]
        n = self.config["n_seeds"]

        self.current_models = [
            lgb.Booster(model_file=f"{model_dir}/current_seed{i}.txt") for i in range(n)
        ]
        self.stage1_models = [
            lgb.Booster(model_file=f"{model_dir}/stage1_seed{i}.txt") for i in range(n)
        ]
        self.stage2_models = [
            lgb.Booster(model_file=f"{model_dir}/stage2_seed{i}.txt") for i in range(n)
        ]

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure column order and categorical dtypes match training."""
        X = X[self.feature_cols].copy()
        for col in self.categorical_features:
            X[col] = X[col].astype("category")
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns an (n_rows, 3) array of probabilities:
        column 0 = P(away win), column 1 = P(draw), column 2 = P(home win)
        """
        Xp = self._prep(X)

        # --- Current (single 3-class) ensemble ---
        current_preds = [m.predict(Xp) for m in self.current_models]
        current_avg = np.mean(current_preds, axis=0)

        # --- Two-stage ensemble ---
        p_draw_preds = [m.predict(Xp) for m in self.stage1_models]
        p_draw_avg = np.mean(p_draw_preds, axis=0)

        p_home_given_not_draw_preds = [m.predict(Xp) for m in self.stage2_models]
        p_home_given_not_draw_avg = np.mean(p_home_given_not_draw_preds, axis=0)

        p_away = (1 - p_draw_avg) * (1 - p_home_given_not_draw_avg)
        p_home = (1 - p_draw_avg) * p_home_given_not_draw_avg
        two_stage_avg = np.column_stack([p_away, p_draw_avg, p_home])

        # --- Final blend ---
        final_pred = self.w_current * current_avg + self.w_two_stage * two_stage_avg
        return final_pred

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Returns hard class predictions: 0=away, 1=draw, 2=home."""
        return np.argmax(self.predict(X), axis=1)


if __name__ == "__main__":
    # Quick smoke test
    model = ModelA("model_a_ensemble")
    print(f"Loaded {len(model.current_models)} seeds per sub-model.")
    print(f"Blend weights: {model.w_current} current / {model.w_two_stage} two-stage")
