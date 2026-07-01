import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

def plot_roc_curves(model_metrics: Dict[str, Dict[str, Any]], template: str = "plotly_dark") -> go.Figure:
    """Plots ROC curves for all compared machine learning models."""
    fig = go.Figure()
    
    colors = {"logistic_regression": "#3498db", "random_forest": "#f1c40f", "gradient_boosting": "#2ecc71"}
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}
    
    for name, metrics in model_metrics.items():
        if "fpr" in metrics and "tpr" in metrics:
            fig.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                name=labels.get(name, name), line=dict(color=colors.get(name, "#7f8c8d"), width=2)
            ))
            
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Guess", line=dict(color="gray", dash="dash")
    ))
    
    fig.update_layout(
        template=template, title="Receiver Operating Characteristic (ROC) Curve Comparison",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_confusion_matrix(cm: List[List[int]], model_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a Confusion Matrix heatmap for a given model classifier."""
    z = np.array(cm)
    x = ["Predicted Defeated", "Predicted Winner"]
    y = ["Actual Defeated", "Actual Winner"]
    
    fig = px.imshow(
        z, text_auto=True, labels=dict(color="SamplesCount"),
        x=x, y=y, color_continuous_scale="Blues",
        title=f"Confusion Matrix Heatmap - {model_name.replace('_', ' ').title()}"
    )
    fig.update_layout(template=template, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, template: str = "plotly_dark") -> go.Figure:
    """Plots a horizontal bar chart of top driving features."""
    fig = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        title=f"Predictive Feature Impacts - {model_name.replace('_', ' ').title()}",
        color="Importance", color_continuous_scale="Blugrn", text="Importance"
    )
    fig.update_layout(
        template=template, yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_model_metrics_comparison(metrics_list: List[Dict[str, Any]], template: str = "plotly_dark") -> go.Figure:
    """Plots a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    df = pd.DataFrame(metrics_list)
    fig = px.bar(
        df, x="Model", y="Score", color="Metric", barmode="group",
        title="Model Accuracy & Classification Performance Comparison",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(template=template, yaxis_range=[0, 1.0], height=400, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_cv_scores(cv_data: Dict[str, List[float]], template: str = "plotly_dark") -> go.Figure:
    """Plots a Box Plot showing Cross-Validation Accuracy Distributions across folds."""
    df_rows = []
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}
    for name, scores in cv_data.items():
        for score in scores:
            df_rows.append({"Model": labels.get(name, name), "Accuracy Score": score})
            
    df = pd.DataFrame(df_rows)
    fig = px.box(
        df, x="Model", y="Accuracy Score", color="Model",
        title="5-Fold Cross-Validation Accuracy Score Variance",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(template=template, height=400, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig
