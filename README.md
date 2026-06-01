# 🏀 NBA Edge: Explainable AI Matchup Predictor

An AI-powered NBA matchup analysis application that predicts game winners, estimates win probabilities, projects total points, and provides human-readable explanations for why a team is favored.

Unlike traditional black-box prediction models, NBA Edge combines machine learning with explainable basketball analytics to help users understand both the prediction and the reasoning behind it.

---

## Features

### Winner Prediction

Predicts the most likely winner of an NBA matchup using machine learning trained on historical NBA game data.

### Win Probability

Provides the probability of the predicted team winning the matchup.

### Confidence Levels

Classifies predictions into:

* Toss-up
* Slight Advantage
* Moderate Advantage
* Strong Advantage

This helps users understand how certain the model is about its prediction.

### Explainable AI Insights

Instead of displaying raw model coefficients, the application translates statistical advantages into basketball-focused explanations, such as:

* Recent form
* Shooting efficiency
* Turnover control
* Team strength
* Home/Away performance

### Two-Sided Analysis

The model explains:

* Why the predicted winner has the edge
* Why the opposing team can still win

This creates a more balanced and realistic game analysis.

### Head-to-Head Insights

Displays historical matchup information between the selected teams, including:

* Win-loss record
* Average margin of victory
* Most recent meeting

### Total Points Projection

Projects the expected total points scored in the game.

### Over/Under Analysis

Compares the projected total points against a user-provided betting line and suggests:

* OVER
* UNDER

---

## Machine Learning Pipeline

### Data Collection

Historical NBA game data is collected using the NBA Stats API.

Training Seasons:

* 2021–22
* 2022–23
* 2023–24
* 2024–25

Current Season:

* 2025–26

Only official NBA franchises are included in the dataset. Exhibition teams, All-Star teams, international clubs, and special-event teams are excluded.

---

### Feature Engineering

The model uses pre-game information only.

Examples include:

* Last 5-game win rate
* Last 10-game win rate
* Points scored
* Points allowed
* Point differential
* Field goal percentage
* Three-point percentage
* Free throw percentage
* Rebounds
* Assists
* Turnovers
* Home/Away performance
* Head-to-head trends
* Elo-based team strength ratings

---

### Models

#### Winner Prediction

Algorithms evaluated:

* Logistic Regression
* Random Forest

Current production model:

* Logistic Regression

Performance:

* Accuracy: ~67%
* ROC-AUC: ~0.72

#### Score Prediction

Separate regression models estimate:

* Home team score
* Away team score
* Total projected points

Performance:

* Home Score MAE: ~10 points
* Away Score MAE: ~10 points
* Total Points MAE: ~16 points

---

## Project Structure

```text
nba-winner-predictor/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── nba_logistic_model.pkl
│   ├── home_score_model.pkl
│   ├── away_score_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── collect_data.py
│   ├── build_training_features.py
│   ├── build_current_features.py
│   ├── train_model.py
│   ├── train_score_models.py
│   └── predict.py
│
└── README.md
```

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/Vanz15/nba-winner-predictor.git
cd nba-winner-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Model Limitations

This project is designed as an analytics and machine learning application rather than a betting system.

The model currently does not account for:

* Player injuries
* Starting lineups
* Trades and roster moves
* Coaching changes
* Live sportsbook odds
* Travel schedules
* Breaking news

Predictions should be interpreted as statistical estimates rather than guarantees.

---

## Future Improvements

Planned enhancements include:

* Live injury tracking
* Starting lineup integration
* Team logo visualization
* Interactive matchup dashboards
* Automated sportsbook odds integration
* Playoff-specific prediction models
* Advanced explainability visualizations

---

## Author

**Aivann (Vanz) Martinez**

BS Computer Science
University of the Philippines Baguio

GitHub: https://github.com/Vanz15
