import numpy as np
import pandas as pd

from data_prep import PROCESSED_DIR

OUTCOMES = ["H", "D", "A"]
VAL_SEASONS = [2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024, 2025]   # touched once, in M6


def load_matches():
    return pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["Date"])


# Yield (season, train, val): train on all earlier seasons, validate on one
def walk_forward_splits(df, val_seasons=VAL_SEASONS):
    for season in val_seasons:
        train = df[df["season"] < season]
        val = df[df["season"] == season]
        yield season, train, val


# "H"/"D"/"A" -> 0/1/2
def encode(results):
    return results.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()


# Mean negative log of the probability given to what actually happened
def log_loss(probs, y, eps=1e-15):
    p_true = probs[np.arange(len(y)), y]
    p_true = np.clip(p_true, eps, 1)
    return -np.mean(np.log(p_true))


# Mean squared gap between probabilities and the one-hot outcome
def brier(probs, y):
    onehot = np.eye(3)[y]
    return ((probs - onehot) ** 2).sum(axis=1).mean()


# Baseline 1: historical H/D/A rates
def prior_baseline(train, val):
    rates = train["FTR"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    return np.tile(rates, (len(val), 1))


# Baseline 2: always say Home with certainty
def home_baseline(train, val):
    return np.tile([1, 0, 0], (len(val), 1))


# Baseline 3: Pinnacle closing odds, margin removed (the number to beat)
def bookmaker_baseline(train, val):
    return val[["PSC_pH", "PSC_pD", "PSC_pA"]].to_numpy()


BASELINES = {
    "prior": prior_baseline,
    "always_home": home_baseline,
    "bookmaker": bookmaker_baseline,
}


if __name__ == "__main__":
    df = load_matches()
    rows = []
    for season, train, val in walk_forward_splits(df):
        y = encode(val["FTR"])
        for name, fn in BASELINES.items():
            probs = fn(train, val)
            rows.append({"season": season, "model": name,
                         "log_loss": log_loss(probs, y), "brier": brier(probs, y)})

    results = pd.DataFrame(rows)
    for metric in ["log_loss", "brier"]:
        table = results.pivot(index="model", columns="season", values=metric)
        table["mean"] = table.mean(axis=1)
        print(f"\n{metric}")
        print(table.round(4))
