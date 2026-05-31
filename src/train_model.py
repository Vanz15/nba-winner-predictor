import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_PATH = "data/processed/nba_games_features.csv"
MODEL_PATH = "models/nba_logistic_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["HOME_GAME_DATE"] = pd.to_datetime(df["HOME_GAME_DATE"])
    return df


def split_data(df):
    train = df[df["HOME_GAME_DATE"] < "2024-10-01"]
    test = df[df["HOME_GAME_DATE"] >= "2024-10-01"]

    drop_cols = [
        "HOME_GAME_ID",
        "HOME_GAME_DATE",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_NAME",
        "home_team_win",
    ]

    X_train = train.drop(columns=drop_cols)
    y_train = train["home_team_win"]

    X_test = test.drop(columns=drop_cols)
    y_test = test["home_team_win"]

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("-" * 30)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1:", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    baseline_accuracy = y_test.value_counts(normalize=True).max()
    print("\nNaive baseline accuracy:", round(baseline_accuracy, 4))

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=8,
    )

    logistic_model.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
    evaluate_model("Random Forest", random_forest, X_test, y_test)

    joblib.dump(logistic_model, MODEL_PATH)
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

    print("\nSaved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()