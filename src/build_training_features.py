import pandas as pd

RAW_PATH = "data/raw/nba_games_training_raw.csv"
OUTPUT_PATH = "data/processed/nba_training_features.csv"


def load_raw_games():
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


def prepare_team_game_logs(df):
    cols = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_NAME",
        "MATCHUP",
        "WL",
        "PTS",
        "PTS_ALLOWED",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "REB",
        "AST",
        "TOV",
    ]

    logs = df[cols].copy()

    logs["is_home"] = logs["MATCHUP"].str.contains("vs.").astype(int)
    logs["win"] = (logs["WL"] == "W").astype(int)

    logs["point_diff"] = logs["PTS"] - logs["PTS_ALLOWED"]
    logs["net_rating_simple"] = logs["point_diff"]

    logs = logs.sort_values(["TEAM_ID", "GAME_DATE"])

    return logs


def add_pre_game_rolling_features(logs):
    rolling_cols = [
        "win",
        "PTS",
        "PTS_ALLOWED",
        "point_diff",
        "net_rating_simple",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "REB",
        "AST",
        "TOV",
    ]

    for col in rolling_cols:
        logs[f"{col}_last5"] = (
            logs.groupby("TEAM_ID")[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        logs[f"{col}_last10"] = (
            logs.groupby("TEAM_ID")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
    logs["season_game_number"] = logs.groupby("TEAM_ID").cumcount() + 1

    logs["season_wins_before"] = (
        logs.groupby("TEAM_ID")["win"].cumsum() - logs["win"]
    )

    logs["season_win_pct"] = (
        logs["season_wins_before"] / (logs["season_game_number"] - 1)
    )

    logs["season_win_pct"] = logs["season_win_pct"].fillna(0)
    logs["prev_game_date"] = logs.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    logs["rest_days"] = (logs["GAME_DATE"] - logs["prev_game_date"]).dt.days
    logs["back_to_back"] = (logs["rest_days"] == 1).astype(int)

    return logs


def add_home_away_form(logs):
    logs = logs.sort_values(["TEAM_ID", "GAME_DATE"])

    home_logs = logs[logs["is_home"] == 1].copy()
    away_logs = logs[logs["is_home"] == 0].copy()

    home_logs["home_win_last5"] = (
        home_logs.groupby("TEAM_ID")["win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    away_logs["away_win_last5"] = (
        away_logs.groupby("TEAM_ID")["win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    logs = pd.merge(
        logs,
        home_logs[["GAME_ID", "TEAM_ID", "home_win_last5"]],
        on=["GAME_ID", "TEAM_ID"],
        how="left",
    )

    logs = pd.merge(
        logs,
        away_logs[["GAME_ID", "TEAM_ID", "away_win_last5"]],
        on=["GAME_ID", "TEAM_ID"],
        how="left",
    )

    return logs


def create_matchup_dataset(logs):
    home = logs[logs["is_home"] == 1].copy()
    away = logs[logs["is_home"] == 0].copy()

    home = home.add_prefix("HOME_")
    away = away.add_prefix("AWAY_")

    games = pd.merge(
        home,
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        how="inner",
    )

    games["home_team_win"] = games["HOME_win"]

    return games


def add_difference_features(games):
    feature_pairs = [
        "win_last5",
        "win_last10",
        "PTS_last5",
        "PTS_last10",
        "PTS_ALLOWED_last5",
        "PTS_ALLOWED_last10",
        "point_diff_last5",
        "point_diff_last10",
        "net_rating_simple_last5",
        "net_rating_simple_last10",
        "FG_PCT_last5",
        "FG_PCT_last10",
        "FG3_PCT_last5",
        "FG3_PCT_last10",
        "FT_PCT_last5",
        "FT_PCT_last10",
        "REB_last5",
        "REB_last10",
        "AST_last5",
        "AST_last10",
        "TOV_last5",
        "TOV_last10",
        "rest_days",
        "back_to_back",
        "season_win_pct",
    ]

    for feature in feature_pairs:
        games[f"{feature}_diff"] = games[f"HOME_{feature}"] - games[f"AWAY_{feature}"]

    games["home_away_form_diff"] = (
        games["HOME_home_win_last5"] - games["AWAY_away_win_last5"]
    )

    games["home_court"] = 1

    return games

def add_elo_features(games, k_factor=20, initial_elo=1500):
    games = games.sort_values("HOME_GAME_DATE").copy()

    team_elos = {}

    home_elo_list = []
    away_elo_list = []

    for _, row in games.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        home_elo = team_elos.get(home_team, initial_elo)
        away_elo = team_elos.get(away_team, initial_elo)

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        actual_home = row["home_team_win"]

        team_elos[home_team] = home_elo + k_factor * (actual_home - expected_home)
        team_elos[away_team] = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))

    games["HOME_elo_rating"] = home_elo_list
    games["AWAY_elo_rating"] = away_elo_list
    games["elo_rating_diff"] = games["HOME_elo_rating"] - games["AWAY_elo_rating"]

    return games

def add_head_to_head_features(games):
    games = games.sort_values("HOME_GAME_DATE").copy()

    history = {}
    h2h_win_pct_diffs = []
    h2h_point_diffs = []

    for _, row in games.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        matchup_key = tuple(sorted([home_team, away_team]))

        past_games = history.get(matchup_key, [])

        if not past_games:
            h2h_win_pct_diffs.append(0.0)
            h2h_point_diffs.append(0.0)
        else:
            home_wins = 0
            away_wins = 0
            home_point_diff_total = 0

            for game in past_games:
                if game["winner"] == home_team:
                    home_wins += 1
                elif game["winner"] == away_team:
                    away_wins += 1

                if game["team_a"] == home_team:
                    home_point_diff_total += game["team_a_pts"] - game["team_b_pts"]
                else:
                    home_point_diff_total += game["team_b_pts"] - game["team_a_pts"]

            total_games = len(past_games)

            h2h_win_pct_diffs.append((home_wins - away_wins) / total_games)
            h2h_point_diffs.append(home_point_diff_total / total_games)

        winner = home_team if row["home_team_win"] == 1 else away_team

        history.setdefault(matchup_key, []).append(
            {
                "team_a": home_team,
                "team_b": away_team,
                "team_a_pts": row["HOME_PTS"],
                "team_b_pts": row["AWAY_PTS"],
                "winner": winner,
            }
        )

    games["head_to_head_win_pct_diff"] = h2h_win_pct_diffs
    games["head_to_head_point_diff"] = h2h_point_diffs

    return games

def select_final_columns(games):
    final_cols = [
        "HOME_GAME_ID",
        "HOME_GAME_DATE",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_NAME",
        "HOME_PTS",
        "AWAY_PTS",
        "home_team_win",
        "home_court",
    ]

    diff_cols = [
        col
        for col in games.columns
        if col.endswith("_diff")
        and not col.startswith("HOME_")
        and not col.startswith("AWAY_")
    ]

    final = games[final_cols + diff_cols].copy()
    final = final.sort_values("HOME_GAME_DATE")
    final = final.dropna()

    return final


def main():
    raw = load_raw_games()
    raw = add_opponent_points(raw)

    logs = prepare_team_game_logs(raw)
    logs = add_pre_game_rolling_features(logs)
    logs = add_home_away_form(logs)

    games = create_matchup_dataset(logs)
    games = add_difference_features(games)
    games = add_elo_features(games)
    games = add_head_to_head_features(games)

    final = select_final_columns(games)

    final.to_csv(OUTPUT_PATH, index=False)

    print("Improved pre-game feature dataset created!")
    print("Shape:", final.shape)
    print("Saved to:", OUTPUT_PATH)
    print(final.head())


if __name__ == "__main__":
    main()