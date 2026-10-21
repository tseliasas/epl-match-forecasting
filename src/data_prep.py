import pandas as pd

from download_data import load_all, RAW_DIR

PROCESSED_DIR = RAW_DIR.parent / "processed"

KEEP = [
    "season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC",
    "B365H", "B365D", "B365A",
    "PSCH", "PSCD", "PSCA",
]

def clean(df):
    df = df[KEEP].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_probs(df, prefix):
    h, d, a = f"{prefix}H", f"{prefix}D", f"{prefix}A"
    implied_sum = 1 / df[h] + 1 / df[d] + 1 / df[a]
    df[f"{prefix}_overround"] = implied_sum - 1
    df[f"{prefix}_pH"] = (1 / df[h]) / implied_sum
    df[f"{prefix}_pD"] = (1 / df[d]) / implied_sum
    df[f"{prefix}_pA"] = (1 / df[a]) / implied_sum
    return df

def add_favourite(df):
    odds = df[["B365H", "B365D", "B365A"]]
    df["fav"] = odds.idxmin(axis=1).str[-1]
    df["fav_won"] = df["fav"] == df["FTR"]
    return df

def season_summary(df):
    return df.groupby("season").agg(
        matches=("FTR", "size"),
        b365_margin=("B365_overround", "mean"),
        fav_win_rate=("fav_won", "mean"),
        home_rate=("FTR", lambda s: (s == "H").mean()),
        draw_rate=("FTR", lambda s: (s == "D").mean()),
        away_rate=("FTR", lambda s: (s == "A").mean()),
    ).round(3)


if __name__ == "__main__":
    df = clean(load_all())
    print("Missing values per column:")
    print(df.isna().sum()[lambda s: s > 0])

    df = add_probs(df, "B365")
    df = add_probs(df, "PSC")
    df = add_favourite(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    print(season_summary(df))
