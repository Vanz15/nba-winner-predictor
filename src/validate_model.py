import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix


DATA_PATH = "data/processed/nba_training_features.csv"
MODEL_PATH = "models/nba_logistic_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df["HOME_GAME_DATE"] = pd.to_datetime(df["HOME_GAME_DATE"])

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    test = df[df["HOME_GAME_DATE"] >= "2024-10-01"].copy()

    X_test = test[feature_columns]
    y_test = test["home_team_win"]

    test["home_win_probability"] = model.predict_proba(X_test)[:, 1]
    test["predicted_home_win"] = (test["home_win_probability"] >= 0.5).astype(int)
    test["correct"] = test["predicted_home_win"] == test["home_team_win"]

    accuracy = accuracy_score(y_test, test["predicted_home_win"])

    print("\nValidation Results")
    print("-" * 30)
    print("Test games:", len(test))
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test["predicted_home_win"]))

    print("\nSample Predictions:")
    sample_cols = [
        "HOME_GAME_DATE",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_NAME",
        "home_team_win",
        "predicted_home_win",
        "home_win_probability",
        "correct",
    ]

    print("\nMost Confident Correct Predictions")
    print("-" * 40)

    correct_preds = test[test["correct"] == True]

    print(
        correct_preds.sort_values(
            "home_win_probability",
            ascending=False
        )[sample_cols]
        .head(10)
        .to_string(index=False)
    )

    print("\nMost Confident Wrong Predictions")
    print("-" * 40)

    wrong_preds = test[test["correct"] == False]

    print(
        wrong_preds.sort_values(
            "home_win_probability",
            ascending=False
        )[sample_cols]
        .head(10)
        .to_string(index=False)
    )

    output_path = "data/processed/validation_predictions.csv"
    test[sample_cols].to_csv(output_path, index=False)

    print("\nSaved validation predictions to:", output_path)


if __name__ == "__main__":
    main()