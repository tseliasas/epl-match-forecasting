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
        "is_home": 0,
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

def add_rolling(long, group_cols, window, tag):
    grouped = long.groupby(group_cols)
    for stat in STATS:
        long[f"{stat}_{tag}{window}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return long


# Days since this team's previous match
def add_rest_days(long):
    long["rest_days"] = long.groupby("team")["Date"].diff().dt.days
    return long

START_ELO = 1500
K = 20
HOME_ADV = 60

def expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def add_elo(df):
    ratings = {}
    home_pre, away_pre = [], []
    for row in df.itertuples():
        rh = ratings.get(row.HomeTeam, START_ELO)
        ra = ratings.get(row.AwayTeam, START_ELO)

        home_pre.append(rh)
        away_pre.append(ra)

        e_home = expected(rh + HOME_ADV, ra)
        s_home = {"H": 1, "D": 0.5, "A": 0}[row.FTR]
        change = K * (s_home - e_home)

        ratings[row.HomeTeam] = rh + change
        ratings[row.AwayTeam] = ra - change

    df["elo_home"] = home_pre
    df["elo_away"] = away_pre
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, ratings


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["Date"])
    long = to_long(df)
    for w in WINDOWS:
        long = add_rolling(long, ["team"], w, "all")
        long = add_rolling(long, ["team", "is_home"], w, "venue")
    long = add_rest_days(long)

    cols = ["Date", "is_home", "points", "points_all5", "points_venue5", "rest_days"]
    print(long[long["team"] == "Arsenal"][cols].head(12))
    df, ratings = add_elo(df)
    print(pd.Series(ratings).sort_values(ascending=False).head(6).round(0))

