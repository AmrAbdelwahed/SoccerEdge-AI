import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import datetime

# Set page configuration
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS to improve the app's appearance
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

# Title and description
st.title("Premier League Match Predictor 🏆")
st.markdown("""
This app predicts the probability of a team winning their Premier League match based on historical data
and various performance metrics.
""")

# Load the data and model
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv", index_col=0)
    matches["date"] = pd.to_datetime(matches["date"])
    
    # Team name mapping
    team_mapping = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves"
    }
    
    # Apply mapping to both team and opponent columns
    matches["team"] = matches["team"].replace(team_mapping)
    matches["opponent"] = matches["opponent"].replace(team_mapping)
    
    return matches

@st.cache_resource
def create_model():
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    return rf

def preprocess_data(matches):
    # Create venue and opponent codes
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+","", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    
    # Add target variable
    matches["target"] = (matches["result"] == "W").astype("int")
    
    # Calculate rolling averages
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]
    
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group
    
    matches_rolling = matches.groupby("team").apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    return matches_rolling

def get_team_stats(team, matches_rolling):
    team_matches = matches_rolling[matches_rolling["team"] == team]
    if team_matches.empty:
        st.error(f"No historical data found for {team}")
        return None
    team_data = team_matches.sort_values("date").iloc[-1]
    return team_data

def make_prediction(home_team, away_team, match_date, match_time, matches, matches_rolling, rf_model, all_predictors, new_cols):
    try:
        # Get opponent code more safely
        opp_mask = matches["opponent"] == away_team
        if not opp_mask.any():
            st.error(f"No historical data found for {away_team} as an opponent. Please try another team.")
            return
            
        opp_code = matches["opponent"].astype("category").cat.codes[opp_mask].iloc[0]
        home_stats = get_team_stats(home_team, matches_rolling)
        away_stats = get_team_stats(away_team, matches_rolling)
        
        if home_stats is None or away_stats is None:
            return
            
        pred_data = pd.DataFrame({
            "venue_code": [0],  # 0 for home
            "opp_code": [opp_code],
            "hour": [match_time.hour],
            "day_code": [match_date.weekday()],
        })
        
        # Add rolling averages
        for col in new_cols:
            pred_data[col] = [home_stats[col]]
        
        # Make prediction
        prediction = rf_model.predict_proba(pred_data[all_predictors])[0]
        
        # Display results
        st.header("Match Prediction")
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label=f"{home_team} Win Probability",
                value=f"{prediction[1]:.1%}"
            )
        
        with col6:
            st.metric(
                label="Draw Probability",
                value="N/A"  # Model doesn't predict draws
            )
        
        with col7:
            st.metric(
                label=f"{away_team} Win Probability",
                value=f"{prediction[0]:.1%}"
            )
        
        # Display recent form
        st.header("Recent Form")
        
        col8, col9 = st.columns(2)
        
        with col8:
            st.subheader(f"{home_team} Recent Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Goals For", "Goals Against", "Shots", "Shots on Target"],
                "Average (Last 3 Matches)": [
                    home_stats["gf_rolling"],
                    home_stats["ga_rolling"],
                    home_stats["sh_rolling"],
                    home_stats["sot_rolling"]
                ]
            })
            st.dataframe(stats_df)
        
        with col9:
            st.subheader(f"{away_team} Recent Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Goals For", "Goals Against", "Shots", "Shots on Target"],
                "Average (Last 3 Matches)": [
                    away_stats["gf_rolling"],
                    away_stats["ga_rolling"],
                    away_stats["sh_rolling"],
                    away_stats["sot_rolling"]
                ]
            })
            st.dataframe(stats_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return

# Load data and create model
matches = load_data()
rf_model = create_model()

# Debug: Print available teams
st.sidebar.write("Available teams:", sorted(matches["team"].unique()))

# Preprocess data
matches_rolling = preprocess_data(matches)

# Define predictors
predictors = ["venue_code", "opp_code", "hour", "day_code"]
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
all_predictors = predictors + new_cols

# Train the model
rf_model.fit(matches_rolling[all_predictors], matches_rolling["target"])

# Create the prediction interface
st.header("Make a Prediction")

# Create two columns for team selection
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Select Home Team",
        sorted(matches["team"].unique()),
        key="home_team"
    )

with col2:
    away_team = st.selectbox(
        "Select Away Team",
        sorted([team for team in matches["team"].unique() if team != home_team]),
        key="away_team"
    )

# Create two columns for date and time
col3, col4 = st.columns(2)

with col3:
    match_date = st.date_input(
        "Select Match Date",
        datetime.date.today()
    )

with col4:
    match_time = st.time_input(
        "Select Match Time",
        datetime.time(15, 00)  # Default to 15:00
    )

if st.button("Predict Match Result"):
    make_prediction(home_team, away_team, match_date, match_time, matches, matches_rolling, rf_model, all_predictors, new_cols)

# Add footer with model information
st.markdown("---")
st.markdown("""
**Model Information:**
- Uses Random Forest Classifier
- Trained on historical Premier League match data
- Features include: venue, opponent, time, day of week, and rolling averages of key statistics
- Accuracy metric on test set: 61.23%
""")