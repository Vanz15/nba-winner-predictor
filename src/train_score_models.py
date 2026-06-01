import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/processed/nba_training_features.csv"

HOME_SCORE_MODEL_PATH = "models/home_score_model.pkl"
AWAY_SCORE_MODEL_PATH = "models/away_score_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["HOME_GAME_DATE"] = pd.to_datetime(df["HOME_GAME_DATE"])
    return df


def main():
    df = load_data()

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    train = df[df["HOME_GAME_DATE"] < "2024-10-01"].copy()
    test = df[df["HOME_GAME_DATE"] >= "2024-10-01"].copy()

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    y_home_train = train["HOME_PTS"]
    y_home_test = test["HOME_PTS"]

    y_away_train = train["AWAY_PTS"]
    y_away_test = test["AWAY_PTS"]

    home_score_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
    )

    away_score_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
    )

    home_score_model.fit(X_train, y_home_train)
    away_score_model.fit(X_train, y_away_train)

    home_preds = home_score_model.predict(X_test)
    away_preds = away_score_model.predict(X_test)

    print("\nScore Prediction Validation")
    print("-" * 35)

    print("Home Score MAE:", round(mean_absolute_error(y_home_test, home_preds), 2))
    print("Away Score MAE:", round(mean_absolute_error(y_away_test, away_preds), 2))

    print("Home Score RMSE:", round(mean_squared_error(y_home_test, home_preds) ** 0.5, 2))
    print("Away Score RMSE:", round(mean_squared_error(y_away_test, away_preds) ** 0.5, 2))

    total_actual = y_home_test + y_away_test
    total_pred = home_preds + away_preds

    print("Total Points MAE:", round(mean_absolute_error(total_actual, total_pred), 2))
    print("Total Points RMSE:", round(mean_squared_error(total_actual, total_pred) ** 0.5, 2))

    joblib.dump(home_score_model, HOME_SCORE_MODEL_PATH)
    joblib.dump(away_score_model, AWAY_SCORE_MODEL_PATH)

    print("\nSaved score models.")
    print("Home:", HOME_SCORE_MODEL_PATH)
    print("Away:", AWAY_SCORE_MODEL_PATH)


if __name__ == "__main__":
    main()