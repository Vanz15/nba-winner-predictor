import streamlit as st
import pandas as pd

from src.predict import predict_game


SNAPSHOT_PATH = "data/processed/current_team_snapshots.csv"


st.set_page_config(
    page_title="NBA Matchup Predictor",
    page_icon="🏀",
    layout="wide",
)


@st.cache_data
def load_teams():
    snapshots = pd.read_csv(SNAPSHOT_PATH)
    return sorted(snapshots["TEAM_NAME"].tolist())


st.title("🏀 NBA Matchup Predictor")
st.caption(
    "An explainable NBA prediction app using recent form, team strength, "
    "head-to-head history, and machine learning."
)

teams = load_teams()

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", teams, index=teams.index("San Antonio Spurs"))

with col2:
    away_team = st.selectbox("Away Team", teams, index=teams.index("New York Knicks"))

over_under_line = st.number_input(
    "Optional Over/Under Line",
    min_value=100.0,
    max_value=300.0,
    value=224.5,
    step=0.5,
)

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Matchup", type="primary"):
        result = predict_game(
            home_team=home_team,
            away_team=away_team,
            over_under_line=over_under_line,
        )

        st.divider()

        st.subheader("Prediction Summary")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Predicted Winner", result["predicted_winner"])

        with metric_col2:
            st.metric(
                "Winner Probability",
                f"{result['winner_probability'] * 100:.2f}%",
            )

        with metric_col3:
            st.metric("Confidence", result["confidence_label"])

        st.divider()

        st.subheader(f"Why {result['predicted_winner']} has the edge")

        predicted_factors = result["two_sided_explanation"]["predicted_team_factors"]

        if predicted_factors:
            for factor in predicted_factors[:5]:
                st.success(factor)
        else:
            st.info("No clear statistical edge found for the predicted winner.")

        other_team = (
            result["away_team"]
            if result["predicted_winner"] == result["home_team"]
            else result["home_team"]
        )

        st.subheader(f"Why {other_team} can still win")

        other_factors = result["two_sided_explanation"]["other_team_factors"]

        if other_factors:
            for factor in other_factors[:5]:
                st.warning(factor)
        else:
            st.info(
                f"{other_team} can still win if it outperforms its recent shooting, "
                "limits turnovers, or benefits from lineup and in-game adjustments."
            )

        st.divider()

        st.subheader("Head-to-Head Insight")
        st.write(result["head_to_head_summary"])

        st.divider()

        st.subheader("Total Points Projection")

        total_col1, total_col2, total_col3 = st.columns(3)

        with total_col1:
            st.metric(
                "Projected Total Points",
                f"{result['predicted_total_points']:.1f}",
            )

        with total_col2:
            st.metric("O/U Line", f"{result['over_under_line']:.1f}")

        with total_col3:
            st.metric("Model Pick", result["over_under_pick"])

        st.write(
            "Note: The over/under line is manually provided and is not fetched "
            "from live sportsbook data."
        )

        if result["score_based_winner"] != result["predicted_winner"]:
            st.info(
                "The winner model and score model disagree. The displayed winner is "
                "based on the classification model, which performed better for winner prediction."
            )

        st.divider()

        st.subheader("Model Limitations")
        st.write(
            "This model currently does not account for injuries, confirmed starting lineups, "
            "player availability, trades, coaching changes, betting market odds, or live news."
        )