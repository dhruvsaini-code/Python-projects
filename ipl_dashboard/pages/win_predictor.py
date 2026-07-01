import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components.ui import render_header, kpi_card
from services.win_predictor import get_or_train_predictor, predict_match_outcome, get_feature_importance
from charts.ml import plot_roc_curves, plot_confusion_matrix, plot_feature_importance, plot_model_metrics_comparison, plot_cv_scores
from config.settings import PLOTLY_CONFIG
from utils.pdf_generator import generate_predictor_pdf

def _render_predictions_panel(model_data: Dict[str, Any], plotly_template: str) -> None:
    """Renders the outcome simulator input sliders and prediction gauges."""
    col_in, col_out = st.columns([1, 1])
    with col_in:
        st.markdown('<div class="subheader-custom" style="margin-top:0px;">Specify Current Match State</div>', unsafe_allow_html=True)
        teams = model_data["teams"]
        cities = model_data["cities"]
        
        model_name = st.selectbox("ML Model Pipeline:", ["logistic_regression", "random_forest", "gradient_boosting"], format_func=lambda x: x.replace('_', ' ').title(), key="predictor_model_select")
        
        col_t1, col_t2 = st.columns(2)
        batting_team = col_t1.selectbox("Batting Team (Chasing):", teams, index=0, key="predictor_bat_team")
        bowling_team = col_t2.selectbox("Bowling Team (Defending):", [t for t in teams if t != batting_team], index=0, key="predictor_bowl_team")
        
        city = st.selectbox("Match Host City:", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0, key="predictor_city")
        target_runs = st.number_input("Target Score to Chase:", min_value=1, max_value=300, value=160, step=1, key="predictor_target")
        
        col_score, col_wickets = st.columns(2)
        current_score = col_score.number_input("Current Score (Batting Team):", min_value=0, max_value=300, value=100, step=1, key="predictor_curr_score")
        wickets_fallen = col_wickets.slider("Wickets Fallen:", min_value=0, max_value=9, value=3, key="predictor_wickets")
        
        col_ov, col_bl = st.columns(2)
        overs_completed = col_ov.slider("Completed Overs Bowled:", min_value=0, max_value=19, value=12, key="predictor_overs")
        balls_in_over = col_bl.slider("Balls Bowled in current over:", min_value=0, max_value=5, value=0, key="predictor_balls")
        overs_input = float(overs_completed + balls_in_over / 10.0)
        
    with col_out:
        _render_predictions_output(model_data, model_name, batting_team, bowling_team, city, target_runs, current_score, wickets_fallen, overs_input, plotly_template)

def _render_predictions_output(model_data: Dict[str, Any], model_name: str, bat_t: str, bowl_t: str, city: str, target: int, score: int, wkts: int, overs: float, template: str) -> None:
    """Helper to render prediction gauges, progress bars and context details."""
    st.markdown('<div class="subheader-custom" style="margin-top:0px;">Calculated Probabilities</div>', unsafe_allow_html=True)
    runs_left = target - score
    balls_left = 120 - (int(overs) * 6 + int(round((overs - int(overs)) * 10)))
    
    if runs_left < 0 or (score > 0 and balls_left == 120):
        st.error("Invalid state. Current score cannot exceed target, and runs cannot be scored on 0 balls.")
        return
        
    bat_prob, bowl_prob = predict_match_outcome(model_data, model_name, bat_t, bowl_t, city, target, score, wkts, overs)
    bat_pct, bowl_pct = round(bat_prob * 100, 1), round(bowl_prob * 100, 1)
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=bat_pct, title={'text': f"{bat_t} Win Probability (%)", 'font': {'size': 16}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3498db"}, 'bgcolor': "rgba(0,0,0,0)",
               'steps': [{'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.2)'}, {'range': [40, 60], 'color': 'rgba(241, 196, 15, 0.2)'}, {'range': [60, 100], 'color': 'rgba(46, 204, 113, 0.2)'}]}
    ))
    fig.update_layout(template=template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Progress bars
    st.markdown(
        f"""
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; font-weight:600; color:#3498db;"><span>🏏 {bat_t}</span><span>{bat_pct}%</span></div>
            <div style="background:rgba(255,255,255,0.08); height:12px; border-radius:4px;"><div style="background:#3498db; width:{bat_pct}%; height:12px; border-radius:4px;"></div></div>
        </div>
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; font-weight:600; color:#e74c3c;"><span>🛡️ {bowl_t}</span><span>{bowl_pct}%</span></div>
            <div style="background:rgba(255,255,255,0.08); height:12px; border-radius:4px;"><div style="background:#e74c3c; width:{bowl_pct}%; height:12px; border-radius:4px;"></div></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # PDF summary report exporter trigger
    _export_sim_pdf(bat_t, bowl_t, city, target, score, wkts, overs, model_name, bat_pct, bowl_pct)

def _export_sim_pdf(bat_t: str, bowl_t: str, city: str, target: int, score: int, wkts: int, overs: float, model: str, bat_pct: float, bowl_pct: float) -> None:
    """Helper to generate simulation PDF report."""
    balls_played = int(overs) * 6 + int(round((overs - int(overs)) * 10))
    crr = (score * 6) / balls_played if balls_played > 0 else 0
    rrr = ((target - score) * 6) / (120 - balls_played) if (120 - balls_played) > 0 else 0
    
    sim_data = {
        "batting_team": bat_t, "bowling_team": bowl_t, "city": city,
        "target_runs": target, "current_score": score, "wickets_fallen": wkts,
        "overs_input": overs, "model_name": model, "batting_win_prob": bat_pct,
        "bowling_win_prob": bowl_pct, "confidence_level": "High" if abs(bat_pct-50)*2 >= 60 else "Moderate",
        "crr": crr, "rrr": rrr
    }
    try:
        pdf_bytes = generate_predictor_pdf(sim_data)
        st.download_button("💾 Download Simulation PDF Report", data=pdf_bytes, file_name="IPL_Win_Prediction.pdf", mime="application/pdf")
    except Exception as e:
        st.warning("Failed to compile prediction report PDF.")

def _render_evaluation_panel(model_data: Dict[str, Any], plotly_template: str) -> None:
    """Renders classification scores, ROC metrics and confusion matrix heatmaps."""
    st.markdown('<div class="subheader-custom">Classifier Model Comparison Metrics</div>', unsafe_allow_html=True)
    
    metrics_list = []
    cv_data = {}
    roc_data = {}
    
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        m = model_data[f"{name}_metrics"]
        cv_data[name] = m["cv_scores"]
        roc_data[name] = m
        
        for key in ["accuracy", "precision", "recall", "f1"]:
            metrics_list.append({"Model": name.replace('_', ' ').title(), "Metric": key.capitalize(), "Score": m[key]})
            
    df_metrics = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": f"{model_data['logistic_regression_metrics']['accuracy']*100:.2f}%", "Precision": f"{model_data['logistic_regression_metrics']['precision']*100:.2f}%", "Recall": f"{model_data['logistic_regression_metrics']['recall']*100:.2f}%"},
        {"Model": "Random Forest", "Accuracy": f"{model_data['random_forest_metrics']['accuracy']*100:.2f}%", "Precision": f"{model_data['random_forest_metrics']['precision']*100:.2f}%", "Recall": f"{model_data['random_forest_metrics']['recall']*100:.2f}%"},
        {"Model": "Gradient Boosting", "Accuracy": f"{model_data['gradient_boosting_metrics']['accuracy']*100:.2f}%", "Precision": f"{model_data['gradient_boosting_metrics']['precision']*100:.2f}%", "Recall": f"{model_data['gradient_boosting_metrics']['recall']*100:.2f}%"}
    ])
    st.dataframe(df_metrics, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_model_metrics_comparison(metrics_list, plotly_template), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cv_scores(cv_data, plotly_template), use_container_width=True)
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_roc_curves(roc_data, plotly_template), use_container_width=True)
    with col4:
        eval_model = st.selectbox("Inspect Matrix / Feature Importance for Model:", ["logistic_regression", "random_forest", "gradient_boosting"], key="eval_model_select")
        st.plotly_chart(plot_confusion_matrix(model_data[f"{eval_model}_metrics"]["confusion_matrix"], eval_model, plotly_template), use_container_width=True)
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    importance_df = get_feature_importance(model_data, eval_model)
    st.plotly_chart(plot_feature_importance(importance_df, eval_model, plotly_template), use_container_width=True)

def show_win_predictor_page(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Win Predictor page layout."""
    render_header("IPL WIN PREDICTOR (MACHINE LEARNING)", "Predict live chasing win probabilities using Logistic Regression, Random Forest, & Gradient Boosting")
    
    try:
        model_data = get_or_train_predictor(matches_df, deliveries_df)
        model_loaded = True
    except Exception as e:
        st.error("Error training or loading predictor ML model pipelines.")
        st.exception(e)
        model_loaded = False
        
    if model_loaded:
        tab_pred, tab_eval = st.tabs(["🤖 Outcome Simulator", "📊 Model Evaluation & Features"])
        with tab_pred:
            _render_predictions_panel(model_data, plotly_template)
        with tab_eval:
            _render_evaluation_panel(model_data, plotly_template)
