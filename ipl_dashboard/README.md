# 🏏 IPL Data Analytics & Win Predictor Dashboard

A production-grade, interactive cricket intelligence dashboard and real-time machine learning match outcome predictor built using **Python**, **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn**. This application provides a portfolio-worthy showcase of advanced sports analytics, data visualization, and predictive modeling.

---

## 🌟 Key Features

### 1. 🏠 Home Page & Season Comparisons
- **General KPIs:** Real-time summary figures (total matches, seasons covered, Runs scored, boundaries, and wickets taken) updating dynamically with global filters.
- **Season Comparison tab:** Line and stack-bar charts tracking average runs per match and boundary frequency (Sixes vs Fours) across seasons.
- **CSV Data Exporter:** One-click download of filtered match lists as CSV reports.

### 2. 👥 Enhanced Team Insights
- **Performance Leaderboards:** Win percentages and total match wins sorted dynamically.
- **Winning Streaks:** Calculates chronological highest winning streak and current active form streaks.
- **Best Season Finder:** Dynamically identifies the season in which a team achieved their highest win rate (minimum 5 matches).
- **Home vs Away Splits:** Classifies all matches as Home, Away, or Neutral venues and plots a grouped bar comparison.
- **Form Badges:** Renders chronological win/loss status sequences (e.g., `W W L W W L L W W W`) with hover details (opponents, match dates).
- **PDF Exporter:** Dynamically compiles a beautiful career summary report for any team using ReportLab.

### 3. 🏏 Player Statistics & profiles
- **Orange & Purple Cap Trends:** Season-by-season progression of top run scorers and wicket takers.
- **Advanced Leaderboards:** Rankings for batsman consistency (averages vs. coefficient of variation), boundary ratios, bowler dot-ball rates, and bowler impact scores.
- **Career Scatter Plots:** interactive strike rate vs average and economy vs wickets scatter graphs.
- **Comparison Engine:** Side-by-side batting/bowling metric cards comparing any two players.
- **PDF Resume:** Compile and download full career profiles as PDF files in-memory.

### 4. 🏟️ Venue Characteristics
- **Individual Stadium deep dive:** Analyzes average powerplay (overs 1-6) and death over (overs 16-20) scores, chasing vs defending ratios, and toss-winner correlation.
- **Pitch Difficulty Index:** Tracks runs scored per wicket (identifies batting-friendly vs bowling-friendly venues).
- **Venue Index Comparisons:** Aggregated index ranking table for all grounds hosting at least 5 matches.

### 5. 📊 Advanced Analytics
- **Overs Phase Analysis:** Comparative distributions of runs and wickets across innings phases.
- **Over-by-Over Heatmap:** Visualizes average runs per over (1 to 20) across teams to show scoring acceleration rates.
- **Chase Timeline Simulator:** Select any past match and watch how the batting team's win probability fluctuated over-by-over.

### 6. 🤖 ML Match Win Predictor
- **Multi-Pipeline Prediction:** Compare live chase predictions from **Logistic Regression (Calibrated)** and **Random Forest Classifiers**.
- **Double Percentage Gauges:** Beautiful radial indicators and progress bars representing current chase outcomes.
- **Predictive timeline:** Contextual run rate details and required scoring run rates updated on input adjustments.
- **Predictive Features Impact:** Custom horizontal bar chart plotting the most important features driving predictions.
- **PDF Report Exporter:** Download the current simulation parameters and predicted outcomes as a PDF report.

---

## 🛠️ Technology Stack

- **Core Logic:** Python 3.8+
- **Data Engineering:** Pandas, Numpy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Dashboard Interface:** Streamlit (v1.30.0+)
- **Machine Learning:** Scikit-learn (One-Hot Encoders, Standard Scalers, Logistic Regression, Random Forest Classifier)
- **PDF Compiler:** ReportLab (v5.0.0+)
- **Styling:** CSS3 & HTML5 (custom Glassmorphic dashboard styles)

---

## 📂 Folder Structure

```
ipl_dashboard/
│
├── data/
│   ├── matches.csv              # Automatically downloaded on first launch (~144KB)
│   └── deliveries.csv           # Automatically downloaded on first launch (~18MB)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Handles downloads, cleans data, standardizes team names
│   ├── preprocessing.py         # Formulates vectorized inputs and targets for pipelines
│   ├── team_analysis.py         # Calculates winning streaks, seasons, and home-away splits
│   ├── player_analysis.py       # Aggregates profiles, Cap trends, and advanced scatter plots
│   ├── venue_analysis.py        # Venue comparisons, phase stats, and difficulty ratings
│   ├── win_predictor.py         # ML pipelines, feature importances, and outcome classifiers
│   └── pdf_generator.py         # ReportLab-based PDF report compiler
│
├── models/
│   └── win_prediction_model.pkl # Pickled classifier pipelines (Auto-trained on launch)
│
├── assets/
│   └── custom.css               # Styling rules for Dark/Light theme switching & animations
│
├── app.py                       # Main routing entrypoint of the dashboard
├── requirements.txt             # Project requirements
└── README.md                    # This readme file
```

---

## 🚀 Setup & Local Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection (required on first launch to automatically download datasets from raw mirrors)

### 1. Get the files
Clone this repository or ensure all folder structures are preserved under `ipl_dashboard/`.

### 2. Set up Virtual Environment & Install Requirements
Create a virtual environment and install package dependencies:
```bash
# Navigate to project root
cd ipl_dashboard

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dev server:
```bash
streamlit run app.py
```
On first launch, the app will:
1. Automatically download `matches.csv` and `deliveries.csv` if missing.
2. Train the Logistic Regression and Random Forest win prediction classifiers.
3. Pickle and save the model file to `models/win_prediction_model.pkl`.
4. Open the dashboard in your default browser at `http://localhost:8501`.

---

## 🌐 Deployment (Streamlit Community Cloud)

You can easily host this dashboard for free on Streamlit Community Cloud:

1. Upload the files to a public GitHub repository. (Note: You can omit the `data/` and `models/` folders, as the dashboard automatically downloads the datasets and trains the model on first launch).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **Deploy an app**, then select your repository, branch, and specify `app.py` as the entry point.
4. Click **Deploy!** Your app will be live with a shareable URL in minutes.

---

## 🔮 Future Enhancements

- **Ball-by-Ball Live Timeline:** Expand match simulator to plot probability changes ball-by-ball.
- **Deep Learning Predictions:** Incorporate TensorFlow or PyTorch sequential models (LSTMs) to capture time-series progression.
- **Bowler vs Batsman Matchup Tool:** Compare historical batsman records against specific bowlers.
- **Real-Time Data Scraping:** Connect to Cricket API endpoints to ingest ongoing IPL match data.
