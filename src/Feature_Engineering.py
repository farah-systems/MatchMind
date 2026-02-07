import numpy as np
import pandas as pd


def to_team_level(df):
    home = df[[
        "Date","Season","HomeTeam","AwayTeam","FTR",
        "FTHG","FTAG","HTHG","HTAG","HS","HST","HF","HC","HY","HR",
        "AS","AST","AF","AC","AY","AR"
    ]].copy()

    home = home.rename(columns={
        "HomeTeam":"Team", "AwayTeam":"Opponent",
        "FTHG":"GF", "FTAG":"GA", "HTHG":"HT_GF", "HTAG":"HT_GA",
        "HS":"Shots", "HST":"SoT", "HF":"Fouls", "HC":"Corners", "HY":"Yellows", "HR":"Reds",
        "AS":"OppShots", "AST":"OppSoT", "AF":"OppFouls", "AC":"OppCorners", "AY":"OppYellows", "AR":"OppReds"
    })
    home["is_home"] = 1

    away = df[[
        "Date","Season","HomeTeam","AwayTeam","FTR",
        "FTHG","FTAG","HTHG","HTAG","HS","HST","HF","HC","HY","HR",
        "AS","AST","AF","AC","AY","AR"
    ]].copy()

    away = away.rename(columns={
        "AwayTeam":"Team", "HomeTeam":"Opponent",
        "FTAG":"GF", "FTHG":"GA", "HTAG":"HT_GF", "HTHG":"HT_GA",
        "AS":"Shots", "AST":"SoT", "AF":"Fouls", "AC":"Corners", "AY":"Yellows", "AR":"Reds",
        "HS":"OppShots", "HST":"OppSoT", "HF":"OppFouls", "HC":"OppCorners", "HY":"OppYellows", "HR":"OppReds"
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
    team_df = team_df.sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)
    team_df["GamesPlayed"] = team_df.groupby("Team").cumcount()
    return team_df


def add_rolling_features(team_df, windows=(3, 5, 15), min_periods=1):
    team_df = team_df.sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)

    mean_cols = [
        "GF","GA","Points",
        "Shots","SoT","Corners","Fouls","Yellows","Reds",
        "OppShots","OppSoT","OppCorners","OppFouls","OppYellows","OppReds"
    ]

    for w in windows:
        for col in mean_cols:
            team_df[f"{col}_roll{w}"] = (
                team_df.groupby("Team")[col]
                .apply(lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean())
                .reset_index(level=0, drop=True)
            )

        sot_sum = (
            team_df.groupby("Team")["SoT"]
            .apply(lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum())
            .reset_index(level=0, drop=True)
        )
        shots_sum = (
            team_df.groupby("Team")["Shots"]
            .apply(lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum())
            .reset_index(level=0, drop=True)
        )
        team_df[f"SoT_pct_roll{w}"] = (sot_sum / shots_sum).replace([np.inf, -np.inf], np.nan)

        opp_sot_sum = (
            team_df.groupby("Team")["OppSoT"]
            .apply(lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum())
            .reset_index(level=0, drop=True)
        )
        opp_shots_sum = (
            team_df.groupby("Team")["OppShots"]
            .apply(lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum())
            .reset_index(level=0, drop=True)
        )
        team_df[f"OppSoT_pct_roll{w}"] = (opp_sot_sum / opp_shots_sum).replace([np.inf, -np.inf], np.nan)

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
        "Date_home","Date_away","Season_home","Season_away",
        "Team_home","Team_away","Opponent_home","Opponent_away",
        "FTR_home","FTR_away","Result_home","Result_away",
        "is_home_home","is_home_away",

        # redundant raw opponent stats (duplicates of the other side's raw stats)
        "OppShots_home","OppSoT_home","OppFouls_home","OppCorners_home","OppYellows_home","OppReds_home",
        "OppShots_away","OppSoT_away","OppFouls_away","OppCorners_away","OppYellows_away","OppReds_away",
    ]
    return match_df.drop(columns=[c for c in drop_cols if c in match_df.columns], errors="ignore")
