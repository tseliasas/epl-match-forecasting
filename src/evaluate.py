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
