import numpy as np
import pandas as pd

from data_prep import PROCESSED_DIR

WINDOWS = [5, 10]
STATS = ["points", "gf", "ga", "sot_for", "sot_against"]

def to_long(df):
    home = pd.DataFrame({
        "match_id": df.index,
        "Date": df["Date"],
        "season": df["season"],
        "team": df["HomeTeam"],
        "is_home": 1,
        "gf": df["FTHG"],
        "ga": df["FTAG"],
        "sot_for": df["HST"],
        "sot_against": df["AST"],
    })
    away = pd.DataFrame({
        "match_id": df.index,
        "Date": df["Date"],
        "season": df["season"],
        "team": df["AwayTeam"],
        "is_home": 2,
        "gf": df["FTAG"],
        "ga": df["FTHG"],
        "sot_for": df["AST"],
        "sot_against": df["HST"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long["gf"] > long["ga"], long["gf"] == long["ga"]],
        [3, 1],
        default=0,
    )
    return long.sort_values(["team", "Date"]).reset_index(drop=True)


# Mean of each stat over the previous `window` matches, never including this one
def add_rolling(long, group_cols, window, tag):
    grouped = long.groupby(group_cols)
    for stat in STATS:
        long[f"{stat}_{tag}{window}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return long

