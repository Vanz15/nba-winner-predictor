import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


TRAINING_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
CURRENT_SEASON = "2025-26"

TRAINING_OUTPUT_PATH = "data/raw/nba_games_training_raw.csv"
CURRENT_OUTPUT_PATH = "data/raw/nba_games_current_raw.csv"


def fetch_games_by_season(season: str) -> pd.DataFrame:
    print(f"Fetching games for {season}...")

    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
    )

    games = gamefinder.get_data_frames()[0]

    games["SEASON"] = season

    time.sleep(1)

    return games


def collect_training_data():
    all_games = []

    for season in TRAINING_SEASONS:
        games = fetch_games_by_season(season)
        all_games.append(games)

    training_df = pd.concat(all_games, ignore_index=True)

    training_df.to_csv(TRAINING_OUTPUT_PATH, index=False)

    print("\nTraining data saved!")
    print("Shape:", training_df.shape)
    print("Path:", TRAINING_OUTPUT_PATH)


def collect_current_season_data():
    current_df = fetch_games_by_season(CURRENT_SEASON)

    current_df.to_csv(CURRENT_OUTPUT_PATH, index=False)

    print("\nCurrent season data saved!")
    print("Shape:", current_df.shape)
    print("Path:", CURRENT_OUTPUT_PATH)


def main():
    collect_training_data()
    collect_current_season_data()


if __name__ == "__main__":
    main()