import pandas as pd
import joblib


SNAPSHOT_PATH = "data/processed/current_team_snapshots.csv"
MODEL_PATH = "models/nba_logistic_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_assets():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    snapshots = pd.read_csv(SNAPSHOT_PATH)

    return model, feature_columns, snapshots


def get_team_snapshot(snapshots, team_name):
    team_row = snapshots[snapshots["TEAM_NAME"] == team_name]

    if team_row.empty:
        raise ValueError(f"Team not found: {team_name}")

    return team_row.iloc[0]


def build_matchup_features(home_snapshot, away_snapshot, feature_columns):
    X = pd.DataFrame(
        data=[[0.0] * len(feature_columns)],
        columns=feature_columns,
        dtype=float,
    )

    for col in feature_columns:
        if col == "home_court":
            X.loc[0, col] = 1

        elif col.endswith("_diff"):
            base_feature = col.replace("_diff", "")

            home_value = home_snapshot.get(base_feature, 0)
            away_value = away_snapshot.get(base_feature, 0)

            X.loc[0, col] = home_value - away_value

        else:
            X.loc[0, col] = 0

    return X


def predict_game(home_team, away_team):
    model, feature_columns, snapshots = load_assets()

    home_snapshot = get_team_snapshot(snapshots, home_team)
    away_snapshot = get_team_snapshot(snapshots, away_team)

    X = build_matchup_features(
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
        feature_columns=feature_columns,
    )

    home_win_probability = model.predict_proba(X)[0][1]

    if home_win_probability >= 0.5:
        predicted_winner = home_team
        winner_probability = home_win_probability
    else:
        predicted_winner = away_team
        winner_probability = 1 - home_win_probability

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": predicted_winner,
        "home_win_probability": home_win_probability,
        "winner_probability": winner_probability,
        "features_used": X,
    }


if __name__ == "__main__":
    result = predict_game(
        home_team="Cleveland Cavaliers",
        away_team="Philadelphia 76ers",
    )

    print("\nNBA Game Prediction")
    print("-" * 30)
    print("Home Team:", result["home_team"])
    print("Away Team:", result["away_team"])
    print("Predicted Winner:", result["predicted_winner"])
    print("Home Win Probability:", round(result["home_win_probability"] * 100, 2), "%")
    print("Winner Probability:", round(result["winner_probability"] * 100, 2), "%")