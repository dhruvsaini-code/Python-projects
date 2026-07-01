import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from typing import Dict, Any, Tuple, List
from constants.paths import MODEL_DIR, MODEL_PATH

def _create_preprocessor(cat_features: List[str], num_features: List[str]) -> ColumnTransformer:
    """Creates preprocessing ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features)
        ]
    )

def _evaluate_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Fits a pipeline and evaluates standard classification metrics."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion_matrix": cm.tolist(),
        "cv_scores": cv_scores.tolist()
    }

def train_and_save_model(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, float]:
    """Trains Logistic Regression, Random Forest, and Gradient Boosting pipelines."""
    from services.preprocessing import prepare_win_prediction_data
    ml_df = prepare_win_prediction_data(matches_df, deliveries_df)
    if len(ml_df) == 0:
        raise ValueError("No training data could be prepared.")
        
    X = ml_df.drop(columns=["result"])
    y = ml_df["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = _create_preprocessor(["batting_team", "bowling_team", "city"], ["runs_left", "balls_left", "wickets_left", "target_runs", "crr", "rrr"])
    
    pipelines = {
        "logistic_regression": Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver="liblinear", C=1.0, random_state=42))]),
        "random_forest": Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1))]),
        "gradient_boosting": Pipeline(steps=[("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42))])
    }
    
    model_data: Dict[str, Any] = {
        "teams": sorted(list(set(X["batting_team"].unique()).union(set(X["bowling_team"].unique())))),
        "cities": sorted(X["city"].unique().tolist())
    }
    
    for name, pipe in pipelines.items():
        model_data[name] = pipe
        model_data[f"{name}_metrics"] = _evaluate_pipeline(pipe, X_train, X_test, y_train, y_test)
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
        
    return {k: model_data[f"{k}_metrics"]["accuracy"] for k in pipelines.keys()}

@st.cache_resource(show_spinner="Loading ML Win Predictor Models...")
def get_or_train_predictor(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """Retrieves trained models or triggers training if not found."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model(matches_df, deliveries_df)
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    if "gradient_boosting" not in model_data:
        train_and_save_model(matches_df, deliveries_df)
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
    return model_data

def get_feature_importance(model_data: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """Extracts features coefficients or importances from a model pipeline."""
    pipeline = model_data[model_name]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])
        
    cleaned_names = [name.replace("cat__", "").replace("num__", "") for name in feature_names]
    
    if model_name in ["random_forest", "gradient_boosting"]:
        importances = classifier.feature_importances_
    else:
        importances = np.abs(classifier.coef_[0])
        
    df = pd.DataFrame({"Feature": cleaned_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(12)
    max_val = df["Importance"].max()
    if max_val > 0:
        df["Importance"] = ((df["Importance"] / max_val) * 100).round(1)
    return df

def _extract_balls_left(overs: float) -> Tuple[int, int]:
    """Extracts balls left and balls played from overs count (decimal)."""
    ov = int(overs)
    balls_in_over = int(round((overs - ov) * 10))
    if balls_in_over >= 6:
         ov += 1
         balls_in_over = 0
    balls_played = ov * 6 + balls_in_over
    return max(0, 120 - balls_played), balls_played

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
    """Predicts win probabilities for the chasing and defending teams."""
    runs_left = max(0, target_runs - current_score)
    balls_left, balls_played = _extract_balls_left(overs)
    wickets_left = max(0, 10 - wickets_fallen)
    
    crr = (current_score * 6) / balls_played if balls_played > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    
    input_df = pd.DataFrame([{
        "batting_team": batting_team, "bowling_team": bowling_team, "city": city,
        "runs_left": runs_left, "balls_left": balls_left, "wickets_left": wickets_left,
        "target_runs": target_runs, "crr": crr, "rrr": rrr
    }])
    
    probabilities = model_data[model_name].predict_proba(input_df)[0]
    
    # Overrides for terminal states
    if runs_left == 0:
        return 1.0, 0.0
    if (wickets_left == 0 or balls_left == 0) and runs_left > 0:
        return 0.0, 1.0
        
    return float(probabilities[1]), float(probabilities[0])
