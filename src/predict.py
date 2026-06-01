import pandas as pd
import joblib


SNAPSHOT_PATH = "data/processed/current_team_snapshots.csv"
CURRENT_RAW_PATH = "data/raw/nba_games_current_raw.csv"

MODEL_PATH = "models/nba_logistic_model.pkl"
HOME_SCORE_MODEL_PATH = "models/home_score_model.pkl"
AWAY_SCORE_MODEL_PATH = "models/away_score_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_assets():
    winner_model = joblib.load(MODEL_PATH)
    home_score_model = joblib.load(HOME_SCORE_MODEL_PATH)
    away_score_model = joblib.load(AWAY_SCORE_MODEL_PATH)

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    snapshots = pd.read_csv(SNAPSHOT_PATH)

    current_games = pd.read_csv(CURRENT_RAW_PATH)
    current_games["GAME_DATE"] = pd.to_datetime(current_games["GAME_DATE"])

    return (
        winner_model,
        home_score_model,
        away_score_model,
        feature_columns,
        snapshots,
        current_games,
    )


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

        elif col == "head_to_head_win_pct_diff":
            X.loc[0, col] = float(home_snapshot.get("head_to_head_win_pct_diff", 0))

        elif col == "head_to_head_point_diff":
            X.loc[0, col] = float(home_snapshot.get("head_to_head_point_diff", 0))

        elif col.endswith("_diff"):
            base_feature = col.replace("_diff", "")

            home_value = float(home_snapshot.get(base_feature, 0))
            away_value = float(away_snapshot.get(base_feature, 0))

            X.loc[0, col] = home_value - away_value

    return X


def format_record_from_win_pct(win_pct, games=10):
    wins = round(float(win_pct) * games)
    losses = games - wins
    return f"{wins}-{losses}"


def format_signed_number(value):
    value = float(value)
    if value > 0:
        return f"+{value:.1f}"
    return f"{value:.1f}"


def get_confidence_label(winner_probability):
    if winner_probability < 0.55:
        return "Toss-up"
    elif winner_probability < 0.62:
        return "Slight Advantage"
    elif winner_probability < 0.72:
        return "Moderate Advantage"
    return "Strong Advantage"


def add_factor(
    home_factors,
    away_factors,
    home_team,
    away_team,
    label,
    home_value,
    away_value,
    higher_is_better=True,
    formatter=lambda x: f"{x:.1f}",
):
    home_value = float(home_value)
    away_value = float(away_value)

    if higher_is_better:
        if home_value > away_value:
            home_factors.append(
                f"{label}: {home_team} {formatter(home_value)} vs "
                f"{away_team} {formatter(away_value)}"
            )
        elif away_value > home_value:
            away_factors.append(
                f"{label}: {away_team} {formatter(away_value)} vs "
                f"{home_team} {formatter(home_value)}"
            )
    else:
        if home_value < away_value:
            home_factors.append(
                f"{label}: {home_team} {formatter(home_value)} vs "
                f"{away_team} {formatter(away_value)}"
            )
        elif away_value < home_value:
            away_factors.append(
                f"{label}: {away_team} {formatter(away_value)} vs "
                f"{home_team} {formatter(home_value)}"
            )


def generate_two_sided_explanation(
    home_snapshot,
    away_snapshot,
    home_team,
    away_team,
    predicted_winner,
    winner_probability,
):
    home_factors = []
    away_factors = []

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Elo team strength",
        home_snapshot["elo_rating"],
        away_snapshot["elo_rating"],
        higher_is_better=True,
        formatter=lambda x: f"{x:.0f}",
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Last 10 record",
        home_snapshot["win_last10"],
        away_snapshot["win_last10"],
        higher_is_better=True,
        formatter=lambda x: format_record_from_win_pct(x, 10),
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Last 10 point differential",
        home_snapshot["point_diff_last10"],
        away_snapshot["point_diff_last10"],
        higher_is_better=True,
        formatter=format_signed_number,
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Field goal efficiency",
        home_snapshot["FG_PCT_last10"],
        away_snapshot["FG_PCT_last10"],
        higher_is_better=True,
        formatter=lambda x: f"{x:.1%}",
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Three-point efficiency",
        home_snapshot["FG3_PCT_last10"],
        away_snapshot["FG3_PCT_last10"],
        higher_is_better=True,
        formatter=lambda x: f"{x:.1%}",
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Turnover control",
        home_snapshot["TOV_last10"],
        away_snapshot["TOV_last10"],
        higher_is_better=False,
        formatter=lambda x: f"{x:.1f} turnovers",
    )

    add_factor(
        home_factors,
        away_factors,
        home_team,
        away_team,
        "Home/Away form",
        home_snapshot["home_win_last5"],
        away_snapshot["away_win_last5"],
        higher_is_better=True,
        formatter=lambda x: format_record_from_win_pct(x, 5),
    )

    predicted_team_factors = (
        home_factors if predicted_winner == home_team else away_factors
    )
    other_team_factors = (
        away_factors if predicted_winner == home_team else home_factors
    )

    confidence_label = get_confidence_label(winner_probability)

    if confidence_label in ["Toss-up", "Slight Advantage"]:
        model_note = (
            "This is a close prediction. The model gives the predicted winner only "
            "a small edge, so the other team still has realistic winning factors."
        )
    else:
        model_note = (
            "The model sees a clearer advantage for the predicted winner, but NBA games "
            "can still change because of shooting variance, injuries, rotations, and "
            "late-game execution."
        )

    return {
        "confidence_label": confidence_label,
        "predicted_team_factors": predicted_team_factors,
        "other_team_factors": other_team_factors,
        "model_note": model_note,
    }


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


def predict_game(home_team, away_team, over_under_line=None):
    (
        winner_model,
        home_score_model,
        away_score_model,
        feature_columns,
        snapshots,
        current_games,
    ) = load_assets()

    home_snapshot = get_team_snapshot(snapshots, home_team)
    away_snapshot = get_team_snapshot(snapshots, away_team)

    X = build_matchup_features(
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
        feature_columns=feature_columns,
    )

    home_win_probability = float(winner_model.predict_proba(X)[0][1])

    if home_win_probability >= 0.5:
        predicted_winner = home_team
        winner_probability = home_win_probability
    else:
        predicted_winner = away_team
        winner_probability = 1 - home_win_probability

    predicted_home_score = float(home_score_model.predict(X)[0])
    predicted_away_score = float(away_score_model.predict(X)[0])
    predicted_total_points = predicted_home_score + predicted_away_score

    score_based_winner = (
        home_team if predicted_home_score > predicted_away_score else away_team
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

    two_sided_explanation = generate_two_sided_explanation(
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
        home_team=home_team,
        away_team=away_team,
        predicted_winner=predicted_winner,
        winner_probability=winner_probability,
    )

    head_to_head_summary = get_head_to_head_summary(
        current_games=current_games,
        home_team=home_team,
        away_team=away_team,
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": predicted_winner,
        "home_win_probability": home_win_probability,
        "winner_probability": winner_probability,
        "confidence_label": two_sided_explanation["confidence_label"],
        "two_sided_explanation": two_sided_explanation,
        "head_to_head_summary": head_to_head_summary,
        "predicted_home_score": predicted_home_score,
        "predicted_away_score": predicted_away_score,
        "predicted_total_points": predicted_total_points,
        "score_based_winner": score_based_winner,
        "over_under_line": over_under_line,
        "over_under_pick": over_under_pick,
        "over_under_edge": over_under_edge,
        "features_used": X,
    }


if __name__ == "__main__":
    result = predict_game(
        home_team="Oklahoma City Thunder",
        away_team="Los Angeles Lakers",
        over_under_line=224.5,
    )

    print("\nNBA Game Prediction")
    print("-" * 30)
    print("Home Team:", result["home_team"])
    print("Away Team:", result["away_team"])
    print("Predicted Winner:", result["predicted_winner"])
    print("Winner Probability:", round(result["winner_probability"] * 100, 2), "%")
    print("Home Win Probability:", round(result["home_win_probability"] * 100, 2), "%")

    print("\nPrediction Confidence")
    print("-" * 30)
    print(result["confidence_label"])

    print(f"\nWhy {result['predicted_winner']} has the edge")
    print("-" * 30)
    if result["two_sided_explanation"]["predicted_team_factors"]:
        for factor in result["two_sided_explanation"]["predicted_team_factors"][:5]:
            print(f"- {factor}")
    else:
        print("- No clear statistical edge found for the predicted winner.")

    other_team = (
        result["away_team"]
        if result["predicted_winner"] == result["home_team"]
        else result["home_team"]
    )

    print(f"\nWhy {other_team} can still win")
    print("-" * 30)

    if result["two_sided_explanation"]["other_team_factors"]:
        for factor in result["two_sided_explanation"]["other_team_factors"][:5]:
            print(f"- {factor}")
    else:
        print(
            f"- {other_team} can still win if it outperforms its recent shooting and scoring trends."
        )
        print(
            "- A strong defensive performance or lower-turnover game could reduce the projected gap."
        )
        print(
            "- The model does not account for injuries, lineup changes, or coaching adjustments."
        )

    print("\nModel Note")
    print("-" * 30)
    print(result["two_sided_explanation"]["model_note"])

    print("\nHead-to-Head Insight this Season")
    print("-" * 30)
    print(result["head_to_head_summary"])

    print("\nTotal Points Projection")
    print("-" * 30)
    print(f"Projected Total Points: {result['predicted_total_points']:.1f}")
    # print(
    #     "Note: Team score prediction is currently used only to estimate total points, "
    #     "not to override the winner model."
    # )

    if result["score_based_winner"] != result["predicted_winner"]:
        print(
            "\nModel Consistency Note"
            "\n------------------------------"
            "\nThe winner model and score model disagree. "
            "The displayed winner is based on the classification model, "
            "which performed better for winner prediction."
        )

    if result["over_under_line"] is not None:
        print("\nOver/Under")
        print("-" * 30)
        print("Line:", result["over_under_line"])
        print("Model Pick:", result["over_under_pick"])
        print("Edge:", round(result["over_under_edge"], 1))
        print(
            "Note: The over/under line is manually provided and is not fetched from live sportsbook data. Please check live updates of O/U line."
        )
    
    print("\nModel Limitations")
    print("-" * 30)
    print(
        "This model currently does not account for injuries, confirmed starting lineups, "
        "player availability, trades, coaching changes, betting market odds, or live news."
    )