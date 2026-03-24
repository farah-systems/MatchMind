import pandas as pd


def load_and_merge_seasons(
    seasons,
    data_dir="../data/raw",
    filename_template="EPL {season}.csv",
):
    dfs = []
    for season in seasons:
        path = f"{data_dir}/{filename_template.format(season=season)}"
        d = pd.read_csv(path)
        d["Season"] = season
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def drop_betting_columns(df):
    betting_keywords = [
        "B365","BW","IW","PS","WH","VC","Bb",
        "Max","Avg","AH",">","<","CH","CAH","LB","SJ"
    ]
    cols_to_drop = [c for c in df.columns if any(k in c for k in betting_keywords)]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df.drop(columns=["Div","Time"], errors="ignore")
