from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FIRST_START_YEAR = 2015 # I will use the 2015/16 season as warmup to determine elos for teams, not used for training
LAST_START_YEAR = 2025

def season_code(start_year):
    return f"{start_year % 100:02d}{(start_year % 100) + 1:02d}"

def download_season(code):
    path = RAW_DIR / f"E0_{code}.csv"
    if path.is_file():
        print(f"{code}: already downloaded")
        return path

    response = requests.get(BASE_URL.format(code=code), timeout=15)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"{code}: saved")
    return path

def load_all():
    frames = []
    for year in range(FIRST_START_YEAR, LAST_START_YEAR + 1):
        code = season_code(year)
        path = download_season(code)
        df = pd.read_csv(path)
        df["season"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    print(df.groupby("season").size())   # rows per season
    print("Total rows:", len(df))
