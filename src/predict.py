import pandas as pd
import joblib


SNAPSHOT_PATH = "data/processed/current_team_snapshots.csv"
MODEL_PATH = "models/nba_logistic_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
CURRENT_RAW_PATH = "data/raw/nba_games_current_raw.csv"
HOME_SCORE_MODEL_PATH = "models/home_score_model.pkl"
AWAY_SCORE_MODEL_PATH = "models/away_score_model.pkl"


def load_assets():
    model = joblib.load(MODEL_PATH)
    home_score_model = joblib.load(HOME_SCORE_MODEL_PATH)
    away_score_model = joblib.load(AWAY_SCORE_MODEL_PATH)

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    snapshots = pd.read_csv(SNAPSHOT_PATH)
    current_games = pd.read_csv(CURRENT_RAW_PATH)
    current_games["GAME_DATE"] = pd.to_datetime(current_games["GAME_DATE"])

    return model, home_score_model, away_score_model, feature_columns, snapshots, current_games


def get_team_snapshot(snapshots, team_name):
    team_row = snapshots[snapshots["TEAM_NAME"] == team_name]

    if team_row.empty:
        available = snapshots["TEAM_NAME"].sort_values().tolist()
        raise ValueError(
            f"Team not found: {team_name}\nAvailable teams: {available}"
        )

    return team_row.iloc[0]


def build_matchup_features(home_snapshot, away_snapshot, feature_columns):
    X = pd.DataFrame(
        data=[[0.0] * len(feature_columns)],
        columns=feature_columns,
        dtype=float,
    )

    for col in feature_columns:
        if col == "home_court":
            X.loc[0, col] = 1.0

        elif col == "home_away_form_diff":
            X.loc[0, col] = (
                float(home_snapshot.get("home_win_last5", 0))
                - float(away_snapshot.get("away_win_last5", 0))
            )

        elif col.endswith("_diff"):
            base_feature = col.replace("_diff", "")

            home_value = float(home_snapshot.get(base_feature, 0))
            away_value = float(away_snapshot.get(base_feature, 0))

            X.loc[0, col] = home_value - away_value

    return X

def get_head_to_head_summary(current_games, home_team, away_team):
    games = []

    for game_id in current_games["GAME_ID"].unique():
        game_rows = current_games[current_games["GAME_ID"] == game_id]

        teams_in_game = set(game_rows["TEAM_NAME"].tolist())

        if {home_team, away_team} != teams_in_game:
            continue

        home_row = game_rows[game_rows["MATCHUP"].str.contains("vs.", na=False)]
        away_row = game_rows[game_rows["MATCHUP"].str.contains("@", na=False)]

        if home_row.empty or away_row.empty:
            continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        if pd.isna(home_row["PTS"]) or pd.isna(away_row["PTS"]):
            continue

        game_home_team = home_row["TEAM_NAME"]
        game_away_team = away_row["TEAM_NAME"]

        home_pts = int(home_row["PTS"])
        away_pts = int(away_row["PTS"])

        winner = game_home_team if home_pts > away_pts else game_away_team
        margin = abs(home_pts - away_pts)

        games.append(
            {
                "date": home_row["GAME_DATE"],
                "home": game_home_team,
                "away": game_away_team,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "winner": winner,
                "margin": margin,
            }
        )

    if not games:
        return "These teams have not met yet in the current season."

    home_team_wins = sum(1 for g in games if g["winner"] == home_team)
    away_team_wins = sum(1 for g in games if g["winner"] == away_team)
    avg_margin = sum(g["margin"] for g in games) / len(games)

    latest = sorted(games, key=lambda x: x["date"])[-1]

    return (
        f"Previous meetings this season: {home_team} {home_team_wins}-"
        f"{away_team_wins} vs {away_team}. "
        f"Average margin: {avg_margin:.1f} points. "
        f"Most recent meeting: {latest['away']} {latest['away_pts']} - "
        f"{latest['home']} {latest['home_pts']} at {latest['home']} "
        f"on {latest['date'].date()}, won by {latest['winner']}."
    )

def explain_prediction(model, X):
    scaler = model.named_steps["scaler"]
    logistic_model = model.named_steps["model"]

    X_scaled = scaler.transform(X)
    coefficients = logistic_model.coef_[0]
    contributions = X_scaled[0] * coefficients

    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "value": X.iloc[0].values,
            "contribution": contributions,
        }
    )

    explanation["absolute_contribution"] = explanation["contribution"].abs()

    return explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    )


def format_record_from_win_pct(win_pct, games=10):
    wins = round(win_pct * games)
    losses = games - wins
    return f"{wins}-{losses}"


def format_signed_number(value):
    if value > 0:
        return f"+{value:.1f}"
    return f"{value:.1f}"


def readable_factor(feature, home_snapshot, away_snapshot, home_team, away_team):
    if feature == "win_last10_diff":
        return (
            f"{home_team} is {format_record_from_win_pct(home_snapshot['win_last10'])} "
            f"in its last 10 games, while {away_team} is "
            f"{format_record_from_win_pct(away_snapshot['win_last10'])}."
        )

    if feature == "win_last5_diff":
        return (
            f"{home_team} is {format_record_from_win_pct(home_snapshot['win_last5'], 5)} "
            f"in its last 5 games, while {away_team} is "
            f"{format_record_from_win_pct(away_snapshot['win_last5'], 5)}."
        )

    if feature in ["point_diff_last10_diff", "net_rating_simple_last10_diff"]:
        return (
            f"{home_team} has a last-10 point differential of "
            f"{format_signed_number(home_snapshot['point_diff_last10'])}, while "
            f"{away_team} has {format_signed_number(away_snapshot['point_diff_last10'])}."
        )

    if feature in ["point_diff_last5_diff", "net_rating_simple_last5_diff"]:
        return (
            f"{home_team} has a last-5 point differential of "
            f"{format_signed_number(home_snapshot['point_diff_last5'])}, while "
            f"{away_team} has {format_signed_number(away_snapshot['point_diff_last5'])}."
        )

    if feature == "FG_PCT_last10_diff":
        return (
            f"{home_team} is shooting {home_snapshot['FG_PCT_last10']:.1%} "
            f"from the field over its last 10 games, compared with "
            f"{away_team}'s {away_snapshot['FG_PCT_last10']:.1%}."
        )

    if feature == "FG_PCT_last5_diff":
        return (
            f"{home_team} is shooting {home_snapshot['FG_PCT_last5']:.1%} "
            f"from the field over its last 5 games, compared with "
            f"{away_team}'s {away_snapshot['FG_PCT_last5']:.1%}."
        )

    if feature == "FG3_PCT_last10_diff":
        return (
            f"{home_team} is shooting {home_snapshot['FG3_PCT_last10']:.1%} "
            f"from three over its last 10 games, compared with "
            f"{away_team}'s {away_snapshot['FG3_PCT_last10']:.1%}."
        )

    if feature == "FG3_PCT_last5_diff":
        return (
            f"{home_team} is shooting {home_snapshot['FG3_PCT_last5']:.1%} "
            f"from three over its last 5 games, compared with "
            f"{away_team}'s {away_snapshot['FG3_PCT_last5']:.1%}."
        )

    if feature == "PTS_last10_diff":
        return (
            f"{home_team} averages {home_snapshot['PTS_last10']:.1f} points "
            f"over its last 10 games, while {away_team} averages "
            f"{away_snapshot['PTS_last10']:.1f}."
        )

    if feature == "PTS_last5_diff":
        return (
            f"{home_team} averages {home_snapshot['PTS_last5']:.1f} points "
            f"over its last 5 games, while {away_team} averages "
            f"{away_snapshot['PTS_last5']:.1f}."
        )

    if feature == "PTS_ALLOWED_last10_diff":
        return (
            f"{home_team} allows {home_snapshot['PTS_ALLOWED_last10']:.1f} points "
            f"over its last 10 games, while {away_team} allows "
            f"{away_snapshot['PTS_ALLOWED_last10']:.1f}."
        )

    if feature == "PTS_ALLOWED_last5_diff":
        return (
            f"{home_team} allows {home_snapshot['PTS_ALLOWED_last5']:.1f} points "
            f"over its last 5 games, while {away_team} allows "
            f"{away_snapshot['PTS_ALLOWED_last5']:.1f}."
        )

    if feature == "TOV_last10_diff":
        return (
            f"{home_team} averages {home_snapshot['TOV_last10']:.1f} turnovers "
            f"over its last 10 games, while {away_team} averages "
            f"{away_snapshot['TOV_last10']:.1f}."
        )

    if feature == "TOV_last5_diff":
        return (
            f"{home_team} averages {home_snapshot['TOV_last5']:.1f} turnovers "
            f"over its last 5 games, while {away_team} averages "
            f"{away_snapshot['TOV_last5']:.1f}."
        )

    if feature == "AST_last10_diff":
        return (
            f"{home_team} averages {home_snapshot['AST_last10']:.1f} assists "
            f"over its last 10 games, while {away_team} averages "
            f"{away_snapshot['AST_last10']:.1f}."
        )

    if feature == "REB_last10_diff":
        return (
            f"{home_team} averages {home_snapshot['REB_last10']:.1f} rebounds "
            f"over its last 10 games, while {away_team} averages "
            f"{away_snapshot['REB_last10']:.1f}."
        )

    if feature == "home_away_form_diff":
        return (
            f"{home_team} is {format_record_from_win_pct(home_snapshot['home_win_last5'], 5)} "
            f"in its last 5 home games, while {away_team} is "
            f"{format_record_from_win_pct(away_snapshot['away_win_last5'], 5)} "
            f"in its last 5 away games."
        )

    return feature.replace("_", " ")


def generate_readable_explanations(
    explanation,
    home_snapshot,
    away_snapshot,
    home_team,
    away_team,
    top_n=5,
):
    readable = []

    for _, row in explanation.head(top_n).iterrows():
        feature = row["feature"]
        contribution = row["contribution"]

        direction = "toward the home team" if contribution > 0 else "toward the away team"

        readable.append(
            {
                "factor": readable_factor(
                    feature,
                    home_snapshot,
                    away_snapshot,
                    home_team,
                    away_team,
                ),
                "direction": direction,
                "impact": abs(contribution),
            }
        )

    return readable


def predict_game(home_team, away_team, over_under_line=None):
    model, home_score_model, away_score_model, feature_columns, snapshots, current_games = load_assets()

    home_snapshot = get_team_snapshot(snapshots, home_team)
    away_snapshot = get_team_snapshot(snapshots, away_team)

    X = build_matchup_features(
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
        feature_columns=feature_columns,
    )

    home_win_probability = model.predict_proba(X)[0][1]
    predicted_home_score = float(home_score_model.predict(X)[0])
    predicted_away_score = float(away_score_model.predict(X)[0])
    predicted_total_points = predicted_home_score + predicted_away_score

    if home_win_probability >= 0.5:
        predicted_winner = home_team
        winner_probability = home_win_probability
    else:
        predicted_winner = away_team
        winner_probability = 1 - home_win_probability

    raw_explanation = explain_prediction(model, X)

    readable_explanations = generate_readable_explanations(
        explanation=raw_explanation,
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
        home_team=home_team,
        away_team=away_team,
        top_n=5,
    )

    head_to_head_summary = get_head_to_head_summary(
        current_games=current_games,
        home_team=home_team,
        away_team=away_team,
    )

    over_under_pick = None
    over_under_edge = None

    if over_under_line is not None:
        over_under_edge = predicted_total_points - over_under_line

        if predicted_total_points > over_under_line:
            over_under_pick = "OVER"
        elif predicted_total_points < over_under_line:
            over_under_pick = "UNDER"
        else:
            over_under_pick = "PUSH"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": predicted_winner,
        "home_win_probability": home_win_probability,
        "winner_probability": winner_probability,
        "features_used": X,
        "raw_explanation": raw_explanation,
        "readable_explanations": readable_explanations,
        "head_to_head_summary": head_to_head_summary,
        "predicted_home_score": predicted_home_score,
        "predicted_away_score": predicted_away_score,
        "predicted_total_points": predicted_total_points,
        "over_under_line": over_under_line,
        "over_under_pick": over_under_pick,
        "over_under_edge": over_under_edge,
    }


if __name__ == "__main__":
    result = predict_game(
        home_team="San Antonio Spurs",
        away_team="New York Knicks",
        over_under_line=224.5,
    )

    print("\nNBA Game Prediction")
    print("-" * 30)
    print("Home Team:", result["home_team"])
    print("Away Team:", result["away_team"])
    print("Predicted Winner:", result["predicted_winner"])
    print("Home Win Probability:", round(result["home_win_probability"] * 100, 2), "%")
    print("Winner Probability:", round(result["winner_probability"] * 100, 2), "%")

    print("\nWhy this prediction?")
    print("-" * 30)

    for i, reason in enumerate(result["readable_explanations"], start=1):
        print(f"{i}. {reason['factor']}")
        #print(f"   Direction: {reason['direction']}")
        print(f"   Impact score: {reason['impact']:.3f}")

    print("\nHead-to-Head Insight this Season:")
    print("-" * 30)
    print(result["head_to_head_summary"])

    print("\nPredicted Score")
    print("-" * 30)
    print(f"{result['away_team']}: {result['predicted_away_score']:.1f}")
    print(f"{result['home_team']}: {result['predicted_home_score']:.1f}")
    print(f"Predicted Total Points: {result['predicted_total_points']:.1f}")

    if result["over_under_line"] is not None:
        print("\nOver/Under")
        print("-" * 30)
        print("Line:", result["over_under_line"])
        print("Model Pick:", result["over_under_pick"])
        print("Edge:", round(result["over_under_edge"], 1))