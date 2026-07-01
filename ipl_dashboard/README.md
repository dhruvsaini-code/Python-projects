# 🏏 IPL Analytics & Win Predictor Dashboard

A production-grade, interactive sports intelligence dashboard and real-time machine learning match outcomes engine built using **Python**, **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn**. This dashboard features clean architecture, advanced ML comparative models, a Fantasy Cricket Assistant, Monte Carlo Match Simulators, and elegant glassmorphic visuals comparable to professional Tableau/PowerBI templates.

---

## 🌟 Key Features

| Tab Page | Core Capabilities |
| :--- | :--- |
| **🏠 Home Page** | Real-time global stats counters (KPI cards), season comparison dashboards, boundaries stacked trends, and raw match logs CSV downloader. |
| **👥 Team Performance** | Chronological win streak calculators, home vs. away split analyses, head-to-head match histories, recent form HTML badge grids, and ReportLab career PDF exports. |
| **🏏 Player Statistics** | Cap leaderboards (slider ranges), advanced consistency indexes, dot-ball metrics, batting/bowling scatter matrices, strike-rate trends, wicket shares, and side-by-side radar comparisons. |
| **🏟️ Venue Insights** | Powerplay and death average run charts, pitch difficulty index mapping, stadium chase vs. defend biases, and a complete grounds index ranking table. |
| **📊 Advanced Analytics** | Principal Component Analysis (PCA) 2D mapping, K-Means player role clustering, team similarity matrices, Toss-to-Winner Sankey flow diagrams, Sunburst hierarchies, and Parallel Coordinates. |
| **🤖 Win Predictor** | Multi-classifier outcome predictions (Logistic Regression vs. Random Forest vs. Gradient Boosting), probability gauges, ROC curves, Confusion Matrix matrices, feature importance rankings, and CV box plots. |
| **🧙‍♂️ Fantasy Assistant** | Dynamic role classifications (WK, BOWL, AR, BAT), Dream11 optimal squad generators (credit-constrained knapsack solver), and player similarity search. |
| **🎮 Match Simulators** | Over-by-over Monte Carlo match simulations, first innings projected score regressors, season tournament winner calculators, and player auction value estimators. |

---

## 🛠️ Technology Stack

- **Core Engine:** Python 3.8+
- **Data Engineering:** Pandas, Numpy
- **Data Visualizations:** Plotly Express & Plotly Graph Objects
- **Dashboard Interface:** Streamlit (v1.30.0+)
- **Machine Learning:** Scikit-learn (preprocessors, estimators, metrics)
- **PDF Exporter:** ReportLab (v5.0.0+)
- **Styling:** CSS3 & HTML5 (custom Glassmorphic dashboard styles)

---

## 📂 Production Folder Structure

```
ipl_dashboard/
├── app.py                     # Entry point & Page Router
├── requirements.txt           # Package dependencies
│
├── assets/
│   └── custom.css             # Glassmorphic custom CSS styling rules
│
├── data/
│   ├── matches.csv              # Match outcomes dataset (Auto-downloaded)
│   └── deliveries.csv           # Ball-by-ball deliveries dataset (Auto-downloaded)
│
├── config/
│   └── settings.py            # Global styles and theme config variables
│
├── constants/
│   ├── paths.py               # Paths configurations
│   └── teams.py               # Team mappings and color configurations
│
├── components/
│   ├── ui.py                  # Custom HTML cards, loaders, and headers
│   ├── sidebar.py             # Navigation router and filter components
│   └── search.py              # Autocomplete search redirect helper
│
├── services/
│   ├── data_loader.py         # Data cleaning cached functions
│   ├── preprocessing.py       # ML vector preprocessors
│   ├── win_predictor.py       # ML training pipeline and predictions
│   ├── analytics_service.py   # PCA, K-Means clustering, and matrices math
│   ├── fantasy_engine.py      # Dream11 lineup selector and points calculators
│   ├── simulator_service.py   # Monte Carlo simulators and score predictors
│   └── valuation_service.py   # Player similarities and auction valuators
│
├── charts/
│   ├── team.py                # Team stats Plotly graphs
│   ├── player.py              # Player cap trends and progression lines
│   ├── venue.py               # Stadium averages and chase ratios
│   ├── advanced.py            # Advanced charts (Sankey, Sunburst, PCA scatter)
│   └── ml.py                  # ROC, Confusion Matrix, and CV box plots
│
├── pages/
│   ├── home.py                # Home tab visualizer
│   ├── team_analysis.py       # Teams tab visualizer
│   ├── player_analysis.py     # Players tab visualizer
│   ├── venue_analysis.py      # Venues tab visualizer
│   ├── advanced_analytics.py  # Advanced analytics visualizer
│   ├── win_predictor.py       # Machine learning win predictor visualizer
│   ├── fantasy_assistant.py   # Fantasy cricket assistant visualizer
│   └── simulators.py          # Monte Carlo simulator tab visualizer
│
└── utils/
    ├── pdf_generator.py       # PDF report generator (ReportLab)
    └── export_helpers.py      # CSV and Excel data exporters
```

---

## 🚀 Setup & Local Installation

### Prerequisites
- Python 3.9+
- Internet connection (required on first launch to automatically download datasets from raw mirrors)

### 1. Set up Virtual Environment & Install Requirements
Create a virtual environment and install dependencies:
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

### 2. Run the Application
Launch the Streamlit dev server:
```bash
streamlit run app.py
```
On first launch, the app will:
1. Automatically download `matches.csv` and `deliveries.csv` if missing in `data/`.
2. Train the Logistic Regression, Random Forest, and Gradient Boosting win prediction classifiers.
3. Pickle and save the model file to `models/win_prediction_model.pkl`.
4. Open the dashboard in your default browser at `http://localhost:8501`.

---

## 🌐 Deployment

### Docker Deployment
Build and run the container locally:
```bash
# Build Docker image
docker build -t ipl-dashboard .

# Run Docker container
docker run -p 8501:8501 ipl-dashboard
```

### Streamlit Community Cloud
1. Upload the folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and select **Deploy an app**.
3. Select your repository, branch, and specify `app.py` as the entrypoint.
4. Click **Deploy!**
