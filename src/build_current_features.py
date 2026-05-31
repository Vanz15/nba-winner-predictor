import pandas as pd

RAW_PATH = "data/raw/nba_games_current_raw.csv"
OUTPUT_PATH = "data/processed/current_team_snapshots.csv"


def load_data():
    df = pd.read_csv(RAW_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def add_opponent_points(df):
    opponent_pts = df[["GAME_ID", "TEAM_ID", "PTS"]].copy()

    opponent_pts = opponent_pts.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "PTS": "PTS_ALLOWED",
        }
    )

    merged = pd.merge(df, opponent_pts, on="GAME_ID", how="inner")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    return merged


def prepare_logs(df):
    df["win"] = (df["WL"] == "W").astype(int)
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    df["point_diff"] = df["PTS"] - df["PTS_ALLOWED"]
    df["net_rating_simple"] = df["point_diff"]

    return df


def build_team_snapshots(df):
    snapshots = []

    for team in df["TEAM_NAME"].unique():
        team_df = df[df["TEAM_NAME"] == team].sort_values("GAME_DATE")

        latest_5 = team_df.tail(5)
        latest_10 = team_df.tail(10)

        latest_home_5 = team_df[team_df["is_home"] == 1].tail(5)
        latest_away_5 = team_df[team_df["is_home"] == 0].tail(5)

        snapshot = {
            "TEAM_NAME": team,

            "win_last5": latest_5["win"].mean(),
            "win_last10": latest_10["win"].mean(),

            "PTS_last5": latest_5["PTS"].mean(),
            "PTS_last10": latest_10["PTS"].mean(),

            "PTS_ALLOWED_last5": latest_5["PTS_ALLOWED"].mean(),
            "PTS_ALLOWED_last10": latest_10["PTS_ALLOWED"].mean(),

            "point_diff_last5": latest_5["point_diff"].mean(),
            "point_diff_last10": latest_10["point_diff"].mean(),

            "net_rating_simple_last5": latest_5["net_rating_simple"].mean(),
            "net_rating_simple_last10": latest_10["net_rating_simple"].mean(),

            "FG_PCT_last5": latest_5["FG_PCT"].mean(),
            "FG_PCT_last10": latest_10["FG_PCT"].mean(),

            "FG3_PCT_last5": latest_5["FG3_PCT"].mean(),
            "FG3_PCT_last10": latest_10["FG3_PCT"].mean(),

            "FT_PCT_last5": latest_5["FT_PCT"].mean(),
            "FT_PCT_last10": latest_10["FT_PCT"].mean(),

            "REB_last5": latest_5["REB"].mean(),
            "REB_last10": latest_10["REB"].mean(),

            "AST_last5": latest_5["AST"].mean(),
            "AST_last10": latest_10["AST"].mean(),

            "TOV_last5": latest_5["TOV"].mean(),
            "TOV_last10": latest_10["TOV"].mean(),

            "home_win_last5": latest_home_5["win"].mean(),
            "away_win_last5": latest_away_5["win"].mean(),
        }

        snapshots.append(snapshot)

    snapshots = pd.DataFrame(snapshots)
    snapshots = snapshots.fillna(0)

    return snapshots


def main():
    df = load_data()
    df = add_opponent_points(df)
    df = prepare_logs(df)

    snapshots = build_team_snapshots(df)

    snapshots.to_csv(OUTPUT_PATH, index=False)

    print("Current team snapshots created!")
    print("Shape:", snapshots.shape)
    print("Saved to:", OUTPUT_PATH)
    print(snapshots.head())


if __name__ == "__main__":
    main()