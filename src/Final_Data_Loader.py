import pandas as pd


BETTING_KEYWORDS = [
    "B365", "BW", "IW", "PS", "WH", "VC", "Bb",
    "Max", "Avg", "AH", "CH", "CAH", "LB", "SJ",">","<"
]


def load_and_merge_seasons(
    seasons,
    data_dir="../data/raw",
    filename_template="EPL {season}.csv",
    verbose=True,
):
    """
    Load multiple season CSV files and merge them into one dataframe.

    Parameters
    ----------
    seasons : list[str]
        Example: ["01-02", "02-03", ..., "22-23"]
    data_dir : str
        Folder containing CSV files
    filename_template : str
        Pattern for filenames
    verbose : bool
        Whether to print loading information

    Returns
    -------
    pd.DataFrame
    """
    dfs = []

    for season in seasons:
        path = f"{data_dir}/{filename_template.format(season=season)}"

        if verbose:
            print(f"Loading {path} ...")

        encodings = ["utf-8", "cp1252", "latin1"]

        for enc in encodings:
            try:
                d = pd.read_csv(
                path,
                on_bad_lines="skip",
                engine="python",
                encoding=enc
                )
            except UnicodeDecodeError:
                continue
        d["Season"] = season
        dfs.append(d)

    if not dfs:
        raise ValueError("No season files were loaded.")

    df = pd.concat(dfs, ignore_index=True)

    if "Date" not in df.columns:
        raise ValueError("Column 'Date' not found in merged dataframe.")

    # Robust date parsing
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    if verbose:
        print("Rows with missing Date after parsing:", df["Date"].isna().sum())

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def drop_betting_columns(df):
    """
    Drop bookmaker / odds columns and other irrelevant raw columns.
    """
    cols_to_drop = [
        c for c in df.columns
        if any(k in c for k in BETTING_KEYWORDS)
    ]

    extra_drop = ["Div", "Time"]

    return df.drop(columns=cols_to_drop + extra_drop, errors="ignore")


def clean_raw_match_data(df):
    """
    Basic raw-data cleaning before feature engineering.
    Keeps only rows with essential match info.
    """
    required = ["Date", "HomeTeam", "AwayTeam", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    df = df.copy()

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_clean_match_data(
    seasons,
    data_dir="../data/raw",
    filename_template="EPL {season}.csv",
    drop_betting=True,
    verbose=True,
    ):
    """
    End-to-end raw data loader.
    """
    df = load_and_merge_seasons(
        seasons=seasons,
        data_dir=data_dir,
        filename_template=filename_template,
        verbose=verbose,
    )

    if drop_betting:
        df = drop_betting_columns(df)

    df = clean_raw_match_data(df)
    return df