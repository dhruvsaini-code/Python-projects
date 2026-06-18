# IPL Data Analysis Dashboard & Win Predictor

A production-quality, interactive cricket data analysis dashboard and real-time machine learning match outcome predictor built using **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn**.

## Features

1. **🏠 Home Page**
   - General overview of IPL history (total matches, seasons, venues, unique players).
   - Interactive data table of matches and outcomes filterable by season.
   - Option to download custom CSV reports.

2. **👥 Team Performance**
   - Overall matches won and Win Rate leaderboard.
   - Season-by-season win and match progression.
   - Toss decision impact and outcome correlation analytics.
   - **Head-to-Head Comparison:** Select any two teams to view head-to-head wins, total contests, and full match logs.

3. **🏏 Player Statistics**
   - Batting and Bowling leaderboards with dynamic thresholds (runs, wickets, strike rates, economy).
   - Season-wise **Orange Cap** (Runs) and **Purple Cap** (Wickets) progression.
   - **Player Search Profile:** Detailed statistics card containing batting and bowling history (career runs, strike rate, average, 50s/100s, wickets, economy, best figures, 3W/5W hauls).

4. **🏟️ Venue Insights**
   - Individual stadium deep dive: average 1st and 2nd innings scores, chasing vs defending success splits, highest/lowest scores, and toss biases.
   - Cross-venue score comparison for top host cities.

5. **📊 Advanced Analytics**
   - Phase-wise runs, wickets, and run rate distribution (**Powerplay**, **Middle**, **Death** overs).
   - Dot ball and boundary percentage ratios by team per season.

6. **🤖 Match Win Predictor**
   - Live match-state outcome predictor during the second innings chase.
   - Supports selecting between **Logistic Regression (Calibrated)** and **Random Forest** models.
   - Dynamic indicators: Plotly Gauge charts, progress bars, and run rate context.

---

## Folder Structure

```
ipl_dashboard/
│
├── data/
│   ├── matches.csv              # Automatically downloaded on first launch
│   └── deliveries.csv           # Automatically downloaded on first launch
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Auto-downloads data & standardizes team names
│   ├── preprocessing.py         # Transforms and aggregates data for models & pages
│   ├── team_analysis.py         # Formulates team statistics and plotly charts
│   ├── player_analysis.py       # Player profiles and seasonal cap trends
│   ├── venue_analysis.py        # Stadium performance, biases, and high/low stats
│   └── win_predictor.py         # Trains scikit-learn models & handles predictions
│
├── models/
│   └── win_prediction_model.pkl # Pickled classifier pipelines (Auto-trained on launch)
│
├── assets/
│   └── custom.css               # Styling rules for UI glassmorphism design
│
├── app.py                       # Application entrypoint & navigation router
├── requirements.txt             # Package dependencies
└── README.md                    # Project instructions (this file)
```

---

## Setup & Local Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection (required on first launch to automatically download datasets from public GitHub mirrors, ~15MB)

### 1. Clone or Copy the project
Ensure all files are placed in the `ipl_dashboard/` directory.

### 2. Install Dependencies
Set up a virtual environment and install requirements:

```bash
# Navigate to the project directory
cd ipl_dashboard

# Create a virtual environment
python -m venv venv

# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Dashboard
Run the Streamlit server locally:

```bash
streamlit run app.py
```
This command will launch a local server. If it is your first time running it, the application will:
1. Automatically download `matches.csv` and `deliveries.csv` if they are missing from the `data/` folder.
2. Train the Scikit-learn Logistic Regression and Random Forest win prediction pipelines.
3. Serialize and save the model to `models/win_prediction_model.pkl`.
4. Open the dashboard interface in your default web browser (usually at `http://localhost:8501`).

---

## Deployment (Streamlit Community Cloud)

You can easily host this dashboard for free on Streamlit Community Cloud:

1. Upload the files to a public GitHub repository. (Note: You can omit the `data/` and `models/` folders, as the dashboard automatically downloads the data and trains/saves the model on its first run).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **Deploy an app**, then select your repository, branch, and specify `app.py` as the entry point.
4. Click **Deploy!** Your app will be live with a shareable URL in minutes.
