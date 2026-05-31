# NBA Winner Predictor

An explainable machine learning project that predicts NBA game outcomes using historical team performance, recent form, home-court advantage, and advanced feature engineering.

## Project Goal

The goal of this project is to build a machine learning system capable of predicting NBA game winners using only information available before tipoff.

Unlike many sports prediction projects that rely on betting odds, this project focuses on interpretable basketball statistics and feature engineering.

## Current Progress

### Phase 1: Data Collection

* Collected NBA game data from NBA Stats API
* Gathered multiple seasons of historical NBA games
* Built automated data collection pipeline

### Phase 2: Feature Engineering

Implemented pre-game features including:

* Last 5 game win percentage
* Last 10 game win percentage
* Rolling point differential
* Rolling net rating
* Shooting efficiency metrics
* Rebounding metrics
* Assist metrics
* Turnover metrics
* Rest day features
* Home vs away performance features

### Phase 3: Exploratory Data Analysis

Key findings:

* Home teams win approximately 55.6% of NBA games.
* Rolling point differential is one of the strongest predictors of game outcomes.
* Recent team form significantly influences winning probability.
* Home and away performance differences provide meaningful predictive value.

## Project Structure

```text
nba-winner-predictor/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
├── notebooks/
├── src/
├── models/
└── screenshots/
```

## Upcoming Work

* Logistic Regression baseline model
* Random Forest model
* XGBoost model
* SHAP explainability
* Streamlit deployment
* Live game prediction interface
