import numpy as np
import pandas as pd


RAW_POSTMATCH_COLS = [
    "GF_home", "GA_home", "HT_GF_home", "HT_GA_home", "Shots_home", "SoT_home",
    "Fouls_home", "Corners_home", "Yellows_home", "Reds_home", "Points_home",
    "GF_away", "GA_away", "HT_GF_away", "HT_GA_away", "Shots_away", "SoT_away",
    "Fouls_away", "Corners_away", "Yellows_away", "Reds_away", "Points_away",
    "FTR", "Result"
]

IDENTIFIER_COLS = [
    "MatchID", "Date", "Season", "HomeTeam", "AwayTeam"
]

FORBIDDEN_FEATURE_COLS = set(RAW_POSTMATCH_COLS + IDENTIFIER_COLS)


def encode_target(match_df, result_col="Result", drop_bad=True):
    """
    Encode target:
    W/H -> 0
    D   -> 1
    L/A -> 2
    """
    match_df = match_df.copy()

    match_df[result_col] = (
        match_df[result_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"W": 0, "D": 1, "L": 2, "H": 0, "A": 2})
    )

    if drop_bad:
        match_df = match_df.dropna(subset=[result_col]).copy()

    match_df[result_col] = match_df[result_col].astype(int)
    return match_df


def assert_no_leakage(feature_cols):
    leaks = [c for c in feature_cols if c in FORBIDDEN_FEATURE_COLS]
    if leaks:
        raise ValueError(f"Leakage columns found in features: {leaks}")


def get_feature_columns(match_df):
    """
    Numeric, pre-match-only features.
    """
    candidate_cols = [c for c in match_df.columns if c not in FORBIDDEN_FEATURE_COLS]
    numeric_cols = match_df[candidate_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

    assert_no_leakage(numeric_cols)
    return numeric_cols


def prepare_xy(df, feature_cols, target_col="Result"):
    assert_no_leakage(feature_cols)

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def make_time_split(match_df, train_end, val_end, date_col="Date"):
    """
    Standard time-based split.
    """
    match_df = match_df.copy()
    match_df = match_df.dropna(subset=[date_col, "Result"]).copy()
    match_df = match_df.sort_values(date_col).reset_index(drop=True)

    train = match_df[match_df[date_col] < train_end].copy()
    val = match_df[(match_df[date_col] >= train_end) & (match_df[date_col] < val_end)].copy()
    test = match_df[match_df[date_col] >= val_end].copy()

    return train, val, test


def make_walk_forward_splits(match_df, min_train_seasons=10, season_col="Season"):
    """
    Walk-forward season-by-season splits.
    Returns list of (train_df, val_df, val_season).
    """
    seasons = sorted(match_df[season_col].dropna().unique().tolist())
    splits = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_df = match_df[match_df[season_col].isin(train_seasons)].copy()
        val_df = match_df[match_df[season_col] == val_season].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        splits.append((train_df, val_df, val_season))

    return splits


def prepare_modeling_data(
    match_df,
    train_end="2021-07-01",
    val_end="2022-07-01",
    target_col="Result",
):
    """
    Full modeling prep:
    - encode target
    - create feature list
    - time split
    - build X/y
    """
    match_df = encode_target(match_df, result_col=target_col, drop_bad=True)
    feature_cols = get_feature_columns(match_df)

    train, val, test = make_time_split(
        match_df,
        train_end=train_end,
        val_end=val_end,
        date_col="Date",
    )

    X_train, y_train = prepare_xy(train, feature_cols, target_col=target_col)
    X_val, y_val = prepare_xy(val, feature_cols, target_col=target_col)
    X_test, y_test = prepare_xy(test, feature_cols, target_col=target_col)

    return {
        "match_df": match_df,
        "feature_cols": feature_cols,
        "train": train,
        "val": val,
        "test": test,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }