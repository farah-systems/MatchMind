"""
dtype_utils.py
==============
Shared memory-optimization helper: shrinks a DataFrame's memory
footprint without changing any value the model actually uses.

- float64 -> float32: LightGBM converts numeric features to float32
  internally during training/inference regardless of what dtype you
  hand it, so this doesn't reduce the precision the model uses -- it
  just stops pandas from storing false extra precision nothing reads.
  float32 keeps ~7 significant decimal digits, which is far more than
  enough for goals, rolling averages, Elo ratings, xG, etc.

- int64 -> smallest safe int dtype: match counts, goal counts, streak
  lengths are nowhere near needing 64 bits.

- low-cardinality object columns (team names, league codes, season
  labels) -> category dtype: these strings repeat thousands of times
  across rows. `category` stores each unique value once plus a small
  integer code per row, instead of a full string per row -- large
  memory savings, and equality/filtering (df["HomeTeam"] == team)
  behaves identically to plain object/string columns.

Deliberately left alone:
  - datetime64 columns (already compact, and downcasting logic below
    doesn't touch them since their dtype name isn't float64/int64/object)
  - high-cardinality object columns (e.g. free-text columns, if any),
    since converting those to category would save little and isn't
    worth the complexity of guessing which ones qualify
"""
import pandas as pd

# Only convert an object column to category if fewer than half its
# values are unique -- avoids wasting a category table on a column
# that's basically all-unique anyway (in this dataset, none should be,
# but this keeps the function safe to reuse elsewhere).
CATEGORY_MAX_UNIQUE_RATIO = 0.5


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast a DataFrame's dtypes in place (returns the same df for chaining)."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dtype == "object":
            n_total = len(df[col])
            if n_total == 0:
                continue
            n_unique = df[col].nunique(dropna=True)
            if (n_unique / n_total) < CATEGORY_MAX_UNIQUE_RATIO:
                df[col] = df[col].astype("category")
    return df
