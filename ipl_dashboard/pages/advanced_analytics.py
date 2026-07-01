import streamlit as st
import pandas as pd
from components.ui import render_header
from services.analytics_service import get_player_pca_clustering, get_correlation_matrix, get_team_similarity_matrix
from services.player_analysis import get_batsman_stats, get_bowler_stats
from charts.advanced import plot_correlation_matrix, plot_pca_clustering, plot_team_similarity, plot_player_bubble_chart, plot_sankey_flow, plot_sunburst_outcomes, plot_treemap_runs, plot_parallel_coordinates

def _render_clustering_tab(filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders PCA and K-Means clustering dashboard for players."""
    st.markdown('<div class="subheader-custom">Player Profile K-Means Clustering</div>', unsafe_allow_html=True)
    c_type = st.radio("Cluster Role Type:", ["Batsmen", "Bowlers"], horizontal=True)
    
    if c_type == "Batsmen":
        bat_df = get_batsman_stats(filtered_deliveries)
        pca_df = get_player_pca_clustering(bat_df, "batsman")
        fig = plot_pca_clustering(pca_df, "batsman", plotly_template)
    else:
        bow_df = get_bowler_stats(filtered_deliveries)
        pca_df = get_player_pca_clustering(bow_df, "bowler")
        fig = plot_pca_clustering(pca_df, "bowler", plotly_template)
        
    st.plotly_chart(fig, use_container_width=True)
    st.write("Clustered Player Categorizations")
    st.dataframe(pca_df[["Player", "Innings", "Average", "Cluster Role"]], use_container_width=True, height=200)

def _render_flow_tab(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders Sankey Flow, Treemap and Sunburst outcome diagrams."""
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_sunburst_outcomes(filtered_matches, plotly_template), use_container_width=True)
    with col2:
        # Construct Sankey Nodes and Links dynamically
        # Node labels: Toss Winners (Unique) -> Decisions (2) -> Winners (Unique)
        tw = sorted(filtered_matches["toss_winner"].unique().tolist())
        w = sorted(filtered_matches["winner"].unique().tolist())
        dec = ["field", "bat"]
        
        nodes = tw + dec + w
        links = []
        
        for _, row in filtered_matches.iterrows():
            if row["toss_winner"] in tw and row["toss_decision"] in dec and row["winner"] in w:
                s1 = tw.index(row["toss_winner"])
                t1 = len(tw) + dec.index(row["toss_decision"])
                links.append((s1, t1))
                
                s2 = t1
                t2 = len(tw) + len(dec) + w.index(row["winner"])
                links.append((s2, t2))
                
        df_links = pd.DataFrame(links, columns=["source", "target"])
        df_grouped = df_links.groupby(["source", "target"]).size().reset_index(name="value")
        
        fig = plot_sankey_flow(df_grouped["source"].tolist(), df_grouped["target"].tolist(), df_grouped["value"].tolist(), nodes, plotly_template)
        st.plotly_chart(fig, use_container_width=True)

def _render_matrix_tab(filtered_matches: pd.DataFrame, plotly_template: str) -> None:
    """Renders Correlation and Team Similarity matrices."""
    col_cl, col_cr = st.columns(2)
    with col_cl:
        corr_df = get_correlation_matrix(filtered_matches)
        st.plotly_chart(plot_correlation_matrix(corr_df, plotly_template), use_container_width=True)
    with col_cr:
        sim_df = get_team_similarity_matrix(filtered_matches)
        st.plotly_chart(plot_team_similarity(sim_df, plotly_template), use_container_width=True)

def _render_bubble_tab(filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders Player Bubble charts and Parallel Coordinates."""
    bat_df = get_batsman_stats(filtered_deliveries)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(plot_player_bubble_chart(bat_df, min_runs=500, template=plotly_template), use_container_width=True)
    with col_r:
        st.plotly_chart(plot_treemap_runs(filtered_deliveries, plotly_template), use_container_width=True)
        
    st.markdown('<hr class="gradient-hr">', unsafe_allow_html=True)
    st.plotly_chart(plot_parallel_coordinates(bat_df[bat_df["Runs"] >= 200], plotly_template), use_container_width=True)

def show_advanced_page(filtered_matches: pd.DataFrame, filtered_deliveries: pd.DataFrame, plotly_template: str) -> None:
    """Renders the entire Advanced page layout."""
    render_header("ADVANCED SPORTS INTELLIGENCE", "Principal Component Analysis, Parallel Coordinates, Sankey flow diagrams, and Sunburst distributions")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 PCA & Clustering", "🔀 Sankey & Sunburst Flows", "📈 Correlation & Team Similarity", "🫧 Bubble & Parallel Coordinates"])
    with tab1:
        _render_clustering_tab(filtered_deliveries, plotly_template)
    with tab2:
        _render_flow_tab(filtered_matches, plotly_template)
    with tab3:
        _render_matrix_tab(filtered_matches, plotly_template)
    with tab4:
        _render_bubble_tab(filtered_deliveries, plotly_template)
