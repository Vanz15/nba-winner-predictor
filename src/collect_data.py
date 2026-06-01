import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


TRAINING_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
CURRENT_SEASON = "2025-26"

TRAINING_OUTPUT_PATH = "data/raw/nba_games_training_raw.csv"
CURRENT_OUTPUT_PATH = "data/raw/nba_games_current_raw.csv"

NBA_TEAMS = {
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
}


def clean_nba_teams_only(df: pd.DataFrame) -> pd.DataFrame:
    before_rows = len(df)

    cleaned_df = df[df["TEAM_NAME"].isin(NBA_TEAMS)].copy()

    after_rows = len(cleaned_df)

    print(f"Removed {before_rows - after_rows} rows from non-NBA teams.")
    print(f"Remaining teams: {cleaned_df['TEAM_NAME'].nunique()}")

    return cleaned_df


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
    training_df = clean_nba_teams_only(training_df)

    training_df.to_csv(TRAINING_OUTPUT_PATH, index=False)

    print("\nTraining data saved!")
    print("Shape:", training_df.shape)
    print("Path:", TRAINING_OUTPUT_PATH)


def collect_current_season_data():
    current_df = fetch_games_by_season(CURRENT_SEASON)
    current_df = clean_nba_teams_only(current_df)

    current_df.to_csv(CURRENT_OUTPUT_PATH, index=False)

    print("\nCurrent season data saved!")
    print("Shape:", current_df.shape)
    print("Path:", CURRENT_OUTPUT_PATH)

    current_df["GAME_DATE"] = pd.to_datetime(current_df["GAME_DATE"])
    latest_date = current_df["GAME_DATE"].max()
    latest_games = current_df[current_df["GAME_DATE"] == latest_date]

    print("\nLatest current-season game date:", latest_date)
    print(latest_games[["GAME_DATE", "MATCHUP", "WL", "PTS"]].head(20))


def main():
    collect_training_data()
    collect_current_season_data()


if __name__ == "__main__":
    main()