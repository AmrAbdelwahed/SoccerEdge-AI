import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Enhanced Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Enhanced Premier League Match Predictor 🏆")
st.markdown("""
This advanced model predicts Premier League match outcomes using XGBoost with multi-class classification
and sophisticated feature engineering.
""")

@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv", index_col=0)
    matches["date"] = pd.to_datetime(matches["date"])
    
    class MissingDict(dict):
        __missing__ = lambda self, key: key
    map_values = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "West Brom": "West Bromwich Albion",
        "Sheffield United": "Sheffield Utd"
    }
    mapping = MissingDict(**map_values)
    matches["team"] = matches["team"].map(mapping)
    matches["opponent"] = matches["opponent"].map(mapping)
    
    return matches

@st.cache_resource
def create_model():
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric=['mlogloss', 'merror'],
    )
    return xgb

def calculate_elo(matches):
    # Initialize Elo ratings
    elo_dict = {}
    k_factor = 32
    
    for team in matches['team'].unique():
        elo_dict[team] = 1500
    
    matches = matches.sort_values('date')
    
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    matches['team_elo'] = 0
    matches['opp_elo'] = 0
    matches['team_elo_diff'] = 0
    
    for idx, match in matches.iterrows():
        team = match['team']
        opponent = match['opponent']
        
        team_elo = elo_dict[team]
        opp_elo = elo_dict[opponent]
        
        matches.at[idx, 'team_elo'] = team_elo
        matches.at[idx, 'opp_elo'] = opp_elo
        matches.at[idx, 'team_elo_diff'] = team_elo - opp_elo
        
        team_expected = expected_score(team_elo, opp_elo)
        
        if match['result'] == 'W':
            team_actual = 1
        elif match['result'] == 'L':
            team_actual = 0
        else:
            team_actual = 0.5
        
        team_new_elo = team_elo + k_factor * (team_actual - team_expected)
        opp_new_elo = opp_elo + k_factor * ((1 - team_actual) - (1 - team_expected))
        
        elo_dict[team] = team_new_elo
        elo_dict[opponent] = opp_new_elo
    
    return matches

def add_form_features(matches):
    def convert_result_to_points(result):
        points = {'W': 3, 'D': 1, 'L': 0}
        return points[result]
    
    matches = matches.sort_values('date')
    matches['result_points'] = matches['result'].apply(convert_result_to_points)
    
    matches['form_points'] = matches.groupby('team')['result_points'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    matches['h2h_wins'] = 0
    matches['h2h_draws'] = 0
    matches['h2h_losses'] = 0
    
    for idx, match in matches.iterrows():
        team = match['team']
        opponent = match['opponent']
        date = match['date']
        
        h2h_history = matches[
            (matches['date'] < date) &
            (((matches['team'] == team) & (matches['opponent'] == opponent)) |
             ((matches['team'] == opponent) & (matches['opponent'] == team)))
        ]
        
        if not h2h_history.empty:
            team_matches = h2h_history[h2h_history['team'] == team]
            matches.at[idx, 'h2h_wins'] = sum(team_matches['result'] == 'W')
            matches.at[idx, 'h2h_draws'] = sum(team_matches['result'] == 'D')
            matches.at[idx, 'h2h_losses'] = sum(team_matches['result'] == 'L')
    
    matches = matches.drop('result_points', axis=1)
    return matches

def preprocess_data(matches):
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+","", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    
    matches["target"] = matches.apply(
        lambda row: 1 if row['result'] == 'W' else (2 if row['result'] == 'D' else 0), 
        axis=1
    )
    
    matches = calculate_elo(matches)
    matches = add_form_features(matches)
    
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    windows = [3, 5, 10]
    
    for window in windows:
        new_cols = [f"{c}_rolling_{window}" for c in cols]
        
        def rolling_averages(group, cols, new_cols):
            group = group.sort_values("date")
            rolling_stats = group[cols].rolling(window, closed='left').mean()
            group[new_cols] = rolling_stats
            return group
        
        matches = matches.groupby("team").apply(
            lambda x: rolling_averages(x, cols, new_cols)
        ).reset_index(drop=True)
    
    for window in windows:
        matches[f'goal_diff_rolling_{window}'] = (
            matches[f'gf_rolling_{window}'] - matches[f'ga_rolling_{window}']
        )
    
    return matches

def get_team_stats(team, matches_rolling):
    team_matches = matches_rolling[matches_rolling["team"] == team].copy()
    if team_matches.empty:
        st.error(f"No historical data found for {team}")
        return None
    return team_matches.sort_values("date").iloc[-1]

def make_prediction(home_team, away_team, match_date, match_time, matches, matches_rolling, xgb_model, all_predictors, scaler, imputer):
    try:
        home_stats = get_team_stats(home_team, matches_rolling)
        away_stats = get_team_stats(away_team, matches_rolling)
        
        if home_stats is None or away_stats is None:
            return
            
        pred_data = pd.DataFrame({
            "venue_code": [0],
            "opp_code": [away_stats.name],
            "hour": [match_time.hour],
            "day_code": [match_date.weekday()],
            "team_elo": [home_stats["team_elo"]],
            "opp_elo": [away_stats["team_elo"]],
            "team_elo_diff": [home_stats["team_elo"] - away_stats["team_elo"]],
            "form_points": [home_stats["form_points"]],
            "h2h_wins": [home_stats["h2h_wins"]],
            "h2h_draws": [home_stats["h2h_draws"]],
            "h2h_losses": [home_stats["h2h_losses"]]
        })
        
        for window in [3, 5, 10]:
            for col in ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]:
                pred_data[f"{col}_rolling_{window}"] = [home_stats[f"{col}_rolling_{window}"]]
            pred_data[f"goal_diff_rolling_{window}"] = [home_stats[f"goal_diff_rolling_{window}"]]
        
        # Handle missing values
        pred_data_imputed = imputer.transform(pred_data[all_predictors])
        pred_data_scaled = scaler.transform(pred_data_imputed)
        
        prediction = xgb_model.predict_proba(pred_data_scaled)[0]
        
        st.header("Match Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"{home_team} Win Probability",
                value=f"{prediction[1]:.1%}"
            )
        
        with col2:
            st.metric(
                label="Draw Probability",
                value=f"{prediction[2]:.1%}"
            )
        
        with col3:
            st.metric(
                label=f"{away_team} Win Probability",
                value=f"{prediction[0]:.1%}"
            )
        
        st.header("Team Analysis")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader(f"{home_team} Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Elo Rating", "Form Points (Last 5)", "Goal Difference (Last 5)", "Head-to-Head Wins"],
                "Value": [
                    f"{home_stats['team_elo']:.0f}",
                    f"{home_stats['form_points']:.0f}",
                    f"{home_stats['goal_diff_rolling_5']:.1f}",
                    f"{home_stats['h2h_wins']:.0f}"
                ]
            })
            st.dataframe(metrics_df)
        
        with col5:
            st.subheader(f"{away_team} Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Elo Rating", "Form Points (Last 5)", "Goal Difference (Last 5)", "Head-to-Head Wins"],
                "Value": [
                    f"{away_stats['team_elo']:.0f}",
                    f"{away_stats['form_points']:.0f}",
                    f"{away_stats['goal_diff_rolling_5']:.1f}",
                    f"{away_stats['h2h_wins']:.0f}"
                ]
            })
            st.dataframe(metrics_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return

# Load and prepare data
matches = load_data()
matches_rolling = preprocess_data(matches)

# Define predictors
basic_predictors = ["venue_code", "opp_code", "hour", "day_code"]
elo_predictors = ["team_elo", "opp_elo", "team_elo_diff"]
form_predictors = ["form_points", "h2h_wins", "h2h_draws", "h2h_losses"]
rolling_predictors = []

for window in [3, 5, 10]:
    for col in ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]:
        rolling_predictors.append(f"{col}_rolling_{window}")
    rolling_predictors.append(f"goal_diff_rolling_{window}")

all_predictors = basic_predictors + elo_predictors + form_predictors + rolling_predictors

# Prepare features and target
X = matches_rolling[all_predictors]
y = matches_rolling.apply(
    lambda row: 1 if row['result'] == 'W' else (2 if row['result'] == 'D' else 0), 
    axis=1
)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create and train model
xgb_model = create_model()
xgb_model.fit(X_train_scaled, y_train)

# Calculate accuracy
train_pred = xgb_model.predict(X_train_scaled)
test_pred = xgb_model.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

# Create prediction interface
st.header("Make a Prediction")

col6, col7 = st.columns(2)

with col6:
    home_team = st.selectbox(
        "Select Home Team",
        sorted(matches["team"].unique()),
        key="home_team"
    )

with col7:
    away_team = st.selectbox(
        "Select Away Team",
        sorted([team for team in matches["team"].unique() if team != home_team]),
        key="away_team"
    )

col8, col9 = st.columns(2)

with col8:
    match_date = st

with col8:
    match_date = st.date_input(
        "Select Match Date",
        datetime.date.today()
    )

with col9:
    match_time = st.time_input(
        "Select Match Time",
        datetime.time(15, 00)
    )

if st.button("Predict Match Result"):
    make_prediction(home_team, away_team, match_date, match_time, matches, matches_rolling, xgb_model, all_predictors, scaler, imputer)

# Add model performance metrics
st.markdown("---")
st.header("Model Performance Metrics")

col10, col11 = st.columns(2)

with col10:
    st.metric(
        label="Training Accuracy",
        value=f"{train_accuracy:.2%}"
    )
    st.markdown("""
    *Training accuracy represents how well the model performs on data it was trained on.*
    """)

with col11:
    st.metric(
        label="Test Accuracy",
        value=f"{test_accuracy:.2%}"
    )
    st.markdown("""
    *Test accuracy represents how well the model performs on new, unseen data.*
    """)

# Add feature importance plot
st.header("Feature Importance Analysis")

feature_importance = pd.DataFrame({
    'feature': all_predictors,
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True).tail(10)

st.bar_chart(feature_importance.set_index('feature'))
st.markdown("""
This chart shows the top 10 most important features the model uses to make predictions.
""")

# Add detailed model information
st.markdown("---")
st.markdown("""
**Enhanced Model Information:**
- Uses XGBoost Classifier with optimized hyperparameters
- Incorporates sophisticated feature engineering:
  - Elo ratings for team strength
  - Multiple rolling average windows (3, 5, and 10 matches)
  - Head-to-head statistics
  - Form points from recent matches
  - Goal difference trends
- Features are standardized using StandardScaler
- Model trained on historical Premier League match data
- Includes both recent form and historical performance metrics
""")

# Add disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:** This model is for informational purposes only. While it uses advanced statistical methods
and historical data to make predictions, football matches are inherently unpredictable and many factors
cannot be captured by statistical models. Please use these predictions responsibly.
""")

# Add usage instructions
st.sidebar.markdown("""
### How to Use
1. Select the home team
2. Select the away team
3. Choose the match date
4. Set the kickoff time
5. Click 'Predict Match Result'

The model will provide:
- Win probabilities for both teams
- Draw probability
- Recent form analysis
- Team performance metrics
""")

# Add feature update information
st.sidebar.markdown("""
### Recent Updates
- Added Elo rating system
- Implemented head-to-head analysis
- Enhanced form calculation
- Added multiple rolling average windows
- Improved visualization of predictions
""")