import numpy as np
import pandas as pd


def to_team_level(df):
    home = df[[
        "Date", "Season", "HomeTeam", "AwayTeam", "FTR",
        "FTHG", "FTAG", "HTHG", "HTAG", "HS", "HST", "HF", "HC", "HY", "HR",
        "AS", "AST", "AF", "AC", "AY", "AR"
    ]].copy()

    home = home.rename(columns={
        "HomeTeam": "Team", "AwayTeam": "Opponent",
        "FTHG": "GF", "FTAG": "GA", "HTHG": "HT_GF", "HTAG": "HT_GA",
        "HS": "Shots", "HST": "SoT", "HF": "Fouls", "HC": "Corners", "HY": "Yellows", "HR": "Reds",
        "AS": "OppShots", "AST": "OppSoT", "AF": "OppFouls", "AC": "OppCorners", "AY": "OppYellows", "AR": "OppReds"
    })
    home["is_home"] = 1

    away = df[[
        "Date", "Season", "HomeTeam", "AwayTeam", "FTR",
        "FTHG", "FTAG", "HTHG", "HTAG", "HS", "HST", "HF", "HC", "HY", "HR",
        "AS", "AST", "AF", "AC", "AY", "AR"
    ]].copy()

    away = away.rename(columns={
        "AwayTeam": "Team", "HomeTeam": "Opponent",
        "FTAG": "GF", "FTHG": "GA", "HTAG": "HT_GF", "HTHG": "HT_GA",
        "AS": "Shots", "AST": "SoT", "AF": "Fouls", "AC": "Corners", "AY": "Yellows", "AR": "Reds",
        "HS": "OppShots", "HST": "OppSoT", "HF": "OppFouls", "HC": "OppCorners", "HY": "OppYellows", "HR": "OppReds"
    })
    away["is_home"] = 0

    team_df = pd.concat([home, away], ignore_index=True)

    team_df["Result"] = np.where(
        (team_df["is_home"] == 1) & (team_df["FTR"] == "H"), "W",
        np.where((team_df["is_home"] == 1) & (team_df["FTR"] == "A"), "L",
        np.where((team_df["is_home"] == 0) & (team_df["FTR"] == "A"), "W",
        np.where((team_df["is_home"] == 0) & (team_df["FTR"] == "H"), "L", "D")))
    )

    team_df["Points"] = team_df["Result"].map({"W": 3, "D": 1, "L": 0}).astype(int)
    return team_df


def add_match_id(team_df):
    team_df = team_df.sort_values(["Date", "Season", "Team", "Opponent", "is_home"]).reset_index(drop=True)
    home_team = np.where(team_df["is_home"].astype(bool), team_df["Team"], team_df["Opponent"])
    away_team = np.where(team_df["is_home"].astype(bool), team_df["Opponent"], team_df["Team"])
    team_df["MatchID"] = (
        team_df["Season"].astype(str) + "_" +
        team_df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        home_team.astype(str) + "_" +
        away_team.astype(str)
    )
    return team_df


def add_games_played(team_df):
    """
    Games already played before current match, by team and season.
    """
    team_df = team_df.sort_values(["Team", "Season", "Date", "MatchID"]).reset_index(drop=True)
    team_df["GamesPlayed"] = team_df.groupby(["Team", "Season"]).cumcount()
    return team_df


def add_rolling_features(team_df, windows=(3, 5, 15), min_periods=1):
    """
    Leakage-free rolling features computed within each team-season.
    """
    team_df = team_df.sort_values(["Team", "Season", "Date", "MatchID"]).reset_index(drop=True)

    mean_cols = [
        "GF", "GA", "Points",
        "Shots", "SoT", "Corners", "Fouls", "Yellows", "Reds",
        "OppShots", "OppSoT", "OppCorners", "OppFouls", "OppYellows", "OppReds"
    ]

    group_cols = ["Team", "Season"]

    for w in windows:
        for col in mean_cols:
            team_df[f"{col}_roll{w}"] = (
                team_df.groupby(group_cols)[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean())
            )

        sot_sum = team_df.groupby(group_cols)["SoT"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum()
        )
        shots_sum = team_df.groupby(group_cols)["Shots"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum()
        )
        team_df[f"SoT_pct_roll{w}"] = (sot_sum / shots_sum).replace([np.inf, -np.inf], np.nan)

        opp_sot_sum = team_df.groupby(group_cols)["OppSoT"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum()
        )
        opp_shots_sum = team_df.groupby(group_cols)["OppShots"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum()
        )
        team_df[f"OppSoT_pct_roll{w}"] = (opp_sot_sum / opp_shots_sum).replace([np.inf, -np.inf], np.nan)

    return team_df


def add_elo_features(
    team_df,
    base_elo=1500.0,
    k=20.0,
    home_advantage=60.0,
):
    """
    Leakage-free Elo at team level.
    Adds pre-match:
    - Elo
    - OppElo
    - EloExp
    """
    team_df = team_df.sort_values(["Date", "MatchID", "is_home"], ascending=[True, True, False]).reset_index(drop=True)

    ratings = {}
    elo_vals = np.zeros(len(team_df), dtype=float)
    opp_elo_vals = np.zeros(len(team_df), dtype=float)
    exp_vals = np.zeros(len(team_df), dtype=float)

    for match_id, idx in team_df.groupby("MatchID", sort=False).groups.items():
        match_rows = team_df.loc[list(idx)].copy()

        if len(match_rows) != 2:
            raise ValueError(f"MatchID {match_id} does not have exactly 2 rows.")

        home_row = match_rows[match_rows["is_home"] == 1]
        away_row = match_rows[match_rows["is_home"] == 0]

        if len(home_row) != 1 or len(away_row) != 1:
            raise ValueError(f"MatchID {match_id} must have exactly one home row and one away row.")

        home_i = home_row.index[0]
        away_i = away_row.index[0]

        home_team = team_df.at[home_i, "Team"]
        away_team = team_df.at[away_i, "Team"]
        ftr = str(team_df.at[home_i, "FTR"]).strip().upper()

        home_elo = ratings.get(home_team, base_elo)
        away_elo = ratings.get(away_team, base_elo)

        home_exp = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0))
        away_exp = 1.0 - home_exp

        elo_vals[home_i] = home_elo
        opp_elo_vals[home_i] = away_elo
        exp_vals[home_i] = home_exp

        elo_vals[away_i] = away_elo
        opp_elo_vals[away_i] = home_elo
        exp_vals[away_i] = away_exp

        if ftr == "H":
            home_score, away_score = 1.0, 0.0
        elif ftr == "A":
            home_score, away_score = 0.0, 1.0
        elif ftr == "D":
            home_score, away_score = 0.5, 0.5
        else:
            raise ValueError(f"Unexpected FTR value {ftr} for MatchID {match_id}")

        ratings[home_team] = home_elo + k * (home_score - home_exp)
        ratings[away_team] = away_elo + k * (away_score - away_exp)

    team_df["Elo"] = elo_vals
    team_df["OppElo"] = opp_elo_vals
    team_df["EloExp"] = exp_vals
    return team_df


def add_additional_features(team_df):
    """
    Additional pre-match team-level features.
    """
    team_df = team_df.sort_values(["Team", "Season", "Date", "MatchID"]).reset_index(drop=True)

    team_df["DaysSinceLastMatch"] = (
        team_df.groupby(["Team", "Season"])["Date"]
        .diff()
        .dt.days
    ).fillna(7)

    team_df["GD_roll5"] = team_df["GF_roll5"] - team_df["GA_roll5"]
    team_df["SeasonProgress"] = team_df["GamesPlayed"] / 38.0

    return team_df


def merge_home_away(team_df):
    home_df = team_df[team_df["is_home"] == 1].copy().add_suffix("_home")
    away_df = team_df[team_df["is_home"] == 0].copy().add_suffix("_away")

    home_df = home_df.rename(columns={"MatchID_home": "MatchID"})
    away_df = away_df.rename(columns={"MatchID_away": "MatchID"})

    match_df = home_df.merge(away_df, on="MatchID", how="left")
    return match_df


def clean_match_df(match_df):
    match_df["HomeTeam"] = match_df["Team_home"]
    match_df["AwayTeam"] = match_df["Team_away"]
    match_df["Date"] = match_df["Date_home"]
    match_df["Season"] = match_df["Season_home"]
    match_df["FTR"] = match_df["FTR_home"]
    match_df["Result"] = match_df["Result_home"]

    drop_cols = [
        "Date_home", "Date_away", "Season_home", "Season_away",
        "Team_home", "Team_away", "Opponent_home", "Opponent_away",
        "FTR_home", "FTR_away", "Result_home", "Result_away",
        "is_home_home", "is_home_away",
        "OppShots_home", "OppSoT_home", "OppFouls_home", "OppCorners_home", "OppYellows_home", "OppReds_home",
        "OppShots_away", "OppSoT_away", "OppFouls_away", "OppCorners_away", "OppYellows_away", "OppReds_away",
    ]
    return match_df.drop(columns=[c for c in drop_cols if c in match_df.columns], errors="ignore")


def add_difference_features(match_df):
    """
    Core matchup difference features.
    """
    match_df = match_df.copy()

    match_df["EloDiff"] = match_df["Elo_home"] - match_df["Elo_away"]

    match_df["FormDiff_roll5"] = (
        match_df["Points_roll5_home"] - match_df["Points_roll5_away"]
    )

    match_df["GD_diff_roll5"] = (
        match_df["GD_roll5_home"] - match_df["GD_roll5_away"]
    )

    match_df["Shots_diff_roll5"] = (
        match_df["Shots_roll5_home"] - match_df["Shots_roll5_away"]
    )

    match_df["FormDiff_roll15"] = (
        match_df["Points_roll15_home"] - match_df["Points_roll15_away"]
    )

    match_df["RestDiff"] = (
        match_df["DaysSinceLastMatch_home"] - match_df["DaysSinceLastMatch_away"]
    )

    return match_df


def add_matchup_features(match_df):
    """
    Attack/defense and shot-based matchup features.
    """
    match_df = match_df.copy()

    match_df["AttackVsDefense_home"] = (
        match_df["GF_roll15_home"] - match_df["GA_roll15_away"]
    )
    match_df["AttackVsDefense_away"] = (
        match_df["GF_roll15_away"] - match_df["GA_roll15_home"]
    )

    match_df["ShotsMatchup_home"] = (
        match_df["Shots_roll15_home"] - match_df["OppShots_roll15_away"]
    )
    match_df["ShotsMatchup_away"] = (
        match_df["Shots_roll15_away"] - match_df["OppShots_roll15_home"]
    )

    match_df["FinishingMatchup_home"] = (
        match_df["SoT_pct_roll15_home"] - match_df["OppSoT_pct_roll15_away"]
    )
    match_df["FinishingMatchup_away"] = (
        match_df["SoT_pct_roll15_away"] - match_df["OppSoT_pct_roll15_home"]
    )

    match_df["HomeBias"] = (
        match_df["Points_roll15_home"] - match_df["Points_roll15_away"]
    )

    return match_df


def add_momentum_features(match_df):
    """
    Trend / momentum features.
    """
    match_df = match_df.copy()

    match_df["FormMomentum_home"] = (
        match_df["Points_roll5_home"] - match_df["Points_roll15_home"]
    )
    match_df["FormMomentum_away"] = (
        match_df["Points_roll5_away"] - match_df["Points_roll15_away"]
    )
    match_df["FormMomentumDiff"] = (
        match_df["FormMomentum_home"] - match_df["FormMomentum_away"]
    )

    match_df["AttackTrend_home"] = (
        match_df["GF_roll5_home"] - match_df["GF_roll15_home"]
    )
    match_df["AttackTrend_away"] = (
        match_df["GF_roll5_away"] - match_df["GF_roll15_away"]
    )

    match_df["DefenseTrend_home"] = (
        match_df["GA_roll5_home"] - match_df["GA_roll15_home"]
    )
    match_df["DefenseTrend_away"] = (
        match_df["GA_roll5_away"] - match_df["GA_roll15_away"]
    )

    return match_df


def drop_nonessential_features(match_df):
    """
    Drop features identified as non-essential after feature selection.
    """
    match_df = match_df.copy()

    drop_cols = [
        "OppReds_roll3_home",
        "OppReds_roll5_home",
        "OppShots_roll15_away",
        "OppCorners_roll5_away",
        "Reds_roll5_away",
        "OppCorners_roll15_away",
        "OppCorners_roll3_home",
        "Fouls_roll5_home",
        "Reds_roll15_away",
        "Shots_roll3_home",
    ]

    return match_df.drop(columns=drop_cols, errors="ignore")


def build_feature_dataset(df):
    """
    Full feature engineering pipeline.
    """
    team_df = to_team_level(df)
    team_df = add_match_id(team_df)
    team_df = add_games_played(team_df)
    team_df = add_rolling_features(team_df)
    team_df = add_elo_features(team_df)
    team_df = add_additional_features(team_df)

    match_df = merge_home_away(team_df)
    match_df = clean_match_df(match_df)
    match_df = add_difference_features(match_df)
    match_df = add_matchup_features(match_df)
    match_df = add_momentum_features(match_df)
    match_df = drop_nonessential_features(match_df)
    return match_df