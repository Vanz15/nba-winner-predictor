import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

def fetch_games_by_season(season:str) -> pd.DataFrame:
    """
    Fetches NBA games for a given season using the nba_api.

    Args:
        season (str): The season for which to fetch games (e.g., '2020-21'
        ).
    """
    print(f"Fetching games for {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00'
    )

    games = gamefinder.get_data_frames()[0]
    time.sleep(1)  # avoid hitting NBA API too quickly

    return games

def main():
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    all_games=[]

    for season in seasons:
        games = fetch_games_by_season(season)
        all_games.append(games)
    
    df = pd.concat(all_games, ignore_index=True)

    print("Shape:", df.shape)
    print(df.head())

    df.to_csv("data/raw/nba_games_raw.csv", index=False)
    print("Saved to data/raw/nba_games_raw.csv")

if __name__ == "__main__":    main()