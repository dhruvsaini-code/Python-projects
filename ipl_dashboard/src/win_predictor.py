import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, List

# Constants
MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "win_prediction_model.pkl")

def train_and_save_model(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, float]:
    """
    Prepares data, trains Logistic Regression and Random Forest models, 
    evaluates them, and saves the trained models to a pickle file.
    """
    from src.preprocessing import prepare_win_prediction_data
    
    print("Preparing win prediction training data...")
    ml_df = prepare_win_prediction_data(matches_df, deliveries_df)
    
    if len(ml_df) == 0:
        raise ValueError("No training data could be prepared. Check matches and deliveries files.")
        
    # Split features and labels
    X = ml_df.drop(columns=["result"])
    y = ml_df["result"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify features
    categorical_features = ["batting_team", "bowling_team", "city"]
    numeric_features = ["runs_left", "balls_left", "wickets_left", "target_runs", "crr", "rrr"]
    
    # Create preprocessing transformer
    # Setting sparse_output=False for dense matrix operations in pipeline
    # handle_unknown='ignore' allows the model to handle new team/city names gracefully
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )
    
    # Define pipelines
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="liblinear", C=1.0, random_state=42))
    ])
    
    # Use max_depth=8 and min_samples_leaf=10 to keep random forest lightweight but accurate
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1))
    ])
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    # Meta lists for input selection filters in UI
    teams = sorted(list(set(X["batting_team"].unique()).union(set(X["bowling_team"].unique()))))
    cities = sorted(X["city"].unique().tolist())
    
    # Save model data
    model_data = {
        "logistic_regression": lr_pipeline,
        "random_forest": rf_pipeline,
        "teams": teams,
        "cities": cities,
        "lr_metrics": {
            "accuracy": lr_acc,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1": lr_f1
        },
        "rf_metrics": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1": rf_f1
        },
        "lr_accuracy": lr_acc,
        "rf_accuracy": rf_acc
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Models saved successfully to {MODEL_PATH}")
    
    return {
        "logistic_regression_accuracy": lr_acc,
        "random_forest_accuracy": rf_acc
    }

@st.cache_resource(show_spinner="Loading ML Win Predictor Models...")
def get_or_train_predictor(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks if model exists, if not trains it, then loads and returns it.
    Uses st.cache_resource to cache loaded ML model pipelines in memory.
    """
    if not os.path.exists(MODEL_PATH):
        print("Pretrained model not found. Starting training...")
        train_and_save_model(matches_df, deliveries_df)
        
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        
    # Backwards compatibility check
    if "lr_metrics" not in model_data:
        # Re-train models to include all precision/recall/f1 metrics
        print("Older model structure detected. Upgrading model metrics...")
        train_and_save_model(matches_df, deliveries_df)
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
            
    return model_data

def get_feature_importance(model_data: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """
    Extracts feature importances or coefficients from the trained Pipeline.
    """
    pipeline = model_data[model_name]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Return generic metrics if names cannot be retrieved
        return pd.DataFrame(columns=["Feature", "Importance"])
        
    cleaned_names = []
    for name in feature_names:
        name = name.replace("cat__", "").replace("num__", "")
        cleaned_names.append(name)
        
    if model_name == "random_forest":
        importances = classifier.feature_importances_
    else:
        # For Logistic Regression, we use the absolute value of coefficients
        importances = np.abs(classifier.coef_[0])
        
    df = pd.DataFrame({
        "Feature": cleaned_names,
        "Importance": importances
    })
    
    # Filter and sort
    df = df.sort_values(by="Importance", ascending=False).head(12)
    
    # Scale relative importance out of 100 for visualization
    max_val = df["Importance"].max()
    if max_val > 0:
        df["Importance"] = ((df["Importance"] / max_val) * 100).round(1)
        
    return df

def predict_match_outcome(
    model_data: Dict[str, Any],
    model_name: str,
    batting_team: str,
    bowling_team: str,
    city: str,
    target_runs: int,
    current_score: int,
    wickets_fallen: int,
    overs: float
) -> Tuple[float, float]:
    """
    Predicts the win probability of the batting team and bowling team 
    given the current match state inputs.
    """
    # Calculate derived inputs matching preprocessing schema
    runs_left = max(0, target_runs - current_score)
    
    # overs is input as a decimal, e.g. 15.2 (15 overs, 2 balls)
    # We need to extract balls bowled and calculate balls left
    ov = int(overs)
    balls_in_over = int(round((overs - ov) * 10))
    # Standard check to keep balls within [0, 5]
    if balls_in_over >= 6:
         ov += 1
         balls_in_over = 0
    balls_played = ov * 6 + balls_in_over
    balls_left = max(0, 120 - balls_played)
    
    wickets_left = max(0, 10 - wickets_fallen)
    
    # Current Run Rate (CRR)
    crr = (current_score * 6) / balls_played if balls_played > 0 else 0.0
    # Required Run Rate (RRR)
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    
    # Build input DataFrame
    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "target_runs": target_runs,
        "crr": crr,
        "rrr": rrr
    }])
    
    # Select pipeline
    pipeline = model_data[model_name]
    
    # Run prediction
    # predict_proba returns [prob_0, prob_1]
    # where 1 is result (batting team wins) and 0 is bowling team wins
    probabilities = pipeline.predict_proba(input_df)[0]
    
    batting_win_prob = float(probabilities[1])
    bowling_win_prob = float(probabilities[0])
    
    # Edge cases overrides
    if runs_left == 0:
        return 1.0, 0.0
    if wickets_left == 0 and runs_left > 0:
        return 0.0, 1.0
    if balls_left == 0 and runs_left > 0:
        return 0.0, 1.0
        
    return batting_win_prob, bowling_win_prob
