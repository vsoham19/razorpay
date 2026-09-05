import os
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add backend/app to path for direct imports inside Streamlit app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app')))

from data_generator import generate_synthetic_data
from rule_engine import run_rule_matching
from ml_engine import build_ml_dataset, train_and_evaluate_ml_models, predict_confidence_for_unmatched
from rag_engine import index_reconciliation_records, answer_settlement_query
from forecaster import run_monte_carlo_cash_forecast
from explainer import build_honest_exception_list, generate_system_explanation

st.set_page_config(
    page_title="Settlement Reconciliation & Forecasting Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Fidelity UI Styling (Google Stitch Enterprise Theme)
st.markdown("""
<style>
    /* Global Container Theme */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Top Header */
    .main-header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.4rem;
    }
    
    /* Executive Metric Cards */
    .metric-card-dark {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: left;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .metric-val-large {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-lbl-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .metric-badge-green {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        margin-left: 6px;
    }
    .metric-badge-orange {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        margin-left: 6px;
    }
    .metric-badge-blue {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        margin-left: 6px;
    }
    
    /* Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        color: #94A3B8;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 600;
        border: 1px solid #334155;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Main Dashboard Top Banner
st.markdown("""
<div class="main-header-container">
    <div class="main-header">⚡ Settlement Reconciliation & Forecasting Copilot</div>
    <div class="sub-header">Razorpay AI Buildathon (Track 04: AI Finance Controller) • High-Fidelity Enterprise Suite</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Config
st.sidebar.markdown("### ⚙️ System Controller")
groq_key_input = st.sidebar.text_input("Groq API Key (Optional)", type="password", help="Pass Groq Key from console.groq.com for live Llama-3.3 LLM responses.")

@st.cache_resource
def initialize_pipeline_cache():
    df_bank, df_ledger, df_truth = generate_synthetic_data(seed=42)
    matched_df, summary = run_rule_matching(df_bank, df_ledger)
    
    X, y = build_ml_dataset(df_bank, df_ledger, df_truth)
    rf_model, scaler, eval_results = train_and_evaluate_ml_models(X, y)
    
    unmatched_df = matched_df[matched_df['match_tier'] == 'TIER_4_UNMATCHED']
    scored_unmatched = predict_confidence_for_unmatched(rf_model, unmatched_df)
    
    index_reconciliation_records(matched_df, scored_unmatched)
    
    forecast_summary, simulated_totals = run_monte_carlo_cash_forecast(df_ledger)
    honest_exceptions = build_honest_exception_list(matched_df, scored_unmatched, forecast_summary)
    
    return {
        "df_bank": df_bank,
        "df_ledger": df_ledger,
        "df_truth": df_truth,
        "matched_df": matched_df,
        "summary": summary,
        "eval_results": eval_results,
        "scored_unmatched": scored_unmatched,
        "forecast_summary": forecast_summary,
        "simulated_totals": simulated_totals,
        "honest_exceptions": honest_exceptions
    }

pipeline_data = initialize_pipeline_cache()

# Layout Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 1. Rule Engine",
    "🤖 2. ML Confidence Scoring",
    "💬 3. Settlement Q&A (RAG)",
    "📈 4. Monte Carlo Forecaster",
    "⚠️ 5. Honest Exceptions & LLM Summary"
])

# ----------------------------------------------------
# TAB 1: RECONCILIATION MATCHING ENGINE
# ----------------------------------------------------
with tab1:
    st.markdown("### Deterministic Multi-Tier Reconciliation Engine")
    summary = pipeline_data["summary"]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">{summary["total_ledger_records"]}</div><div class="metric-lbl-sub">Total Records Processed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">{summary["rule_matched_count"]} <span class="metric-badge-green">Rule Matched</span></div><div class="metric-lbl-sub">Deterministic Matches</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">{summary["rule_match_rate"]}%</div><div class="metric-lbl-sub">Deterministic Match Rate</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">{summary["tier_breakdown"].get("TIER_4_UNMATCHED", 0)} <span class="metric-badge-orange">Action Req.</span></div><div class="metric-lbl-sub">Sent to ML / Exceptions</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("#### Match Tier Distribution")
        tier_df = pd.DataFrame(list(summary["tier_breakdown"].items()), columns=["Match Tier", "Count"])
        fig_pie = px.pie(
            tier_df, names="Match Tier", values="Count", hole=0.5,
            color_discrete_map={
                "TIER_1_EXACT": "#10B981",
                "TIER_2_FEE_ADJUSTED": "#3B82F6",
                "TIER_3_FUZZY_REF": "#8B5CF6",
                "TIER_4_UNMATCHED": "#F59E0B"
            }
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("#### Matched Ledger Records")
        st.dataframe(
            pipeline_data["matched_df"][["ledger_txn_id", "merchant_order_id", "bank_narration", "net_amount", "bank_amount", "amount_diff", "date_gap_days", "match_tier", "confidence_score"]],
            height=360,
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 2: CONFIDENCE SCORING WITH RANDOM FOREST
# ----------------------------------------------------
with tab2:
    st.markdown("### Machine Learning Confidence Scoring")
    eval_res = pipeline_data["eval_results"]
    lr_m = eval_res["baseline_lr"]
    rf_m = eval_res["random_forest"]
    
    st.markdown("#### Held-Out Test Set Performance Benchmark (30% Split)")
    comp_df = pd.DataFrame([
        {"Metric": "Precision", "Logistic Regression (Baseline)": lr_m["precision"], "Random Forest Classifier": rf_m["precision"]},
        {"Metric": "Recall", "Logistic Regression (Baseline)": lr_m["recall"], "Random Forest Classifier": rf_m["recall"]},
        {"Metric": "F1-Score", "Logistic Regression (Baseline)": lr_m["f1"], "Random Forest Classifier": rf_m["f1"]},
        {"Metric": "ROC-AUC", "Logistic Regression (Baseline)": lr_m["roc_auc"], "Random Forest Classifier": rf_m["roc_auc"]}
    ])
    st.table(comp_df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Random Forest Feature Importance Ranking")
        feat_df = pd.DataFrame(list(eval_res["feature_importances"].items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
        fig_feat = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues"
        )
        fig_feat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC')
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with c2:
        st.markdown("#### ML Scored Ambiguous Records (Tier 4)")
        scored_df = pipeline_data["scored_unmatched"]
        st.dataframe(
            scored_df[["ledger_txn_id", "merchant_order_id", "amount_diff", "date_gap_days", "ref_similarity", "ml_confidence_score", "ml_prediction"]],
            height=320,
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 3: SETTLEMENT Q&A AGENT (RAG)
# ----------------------------------------------------
with tab3:
    st.markdown("### Settlement Q&A Agent (ChromaDB Vector Store + RAG)")
    query_input = st.text_input("Enter your natural-language financial query:", value="Why didn't transaction ORD_20260825_1185 reconcile?")
    
    if st.button("Ask Copilot"):
        with st.spinner("Querying ChromaDB vector index..."):
            result = answer_settlement_query(query_input, groq_api_key=groq_key_input)
            
            st.markdown(f"**LLM Engine Active**: `{result['llm_engine']}`")
            st.markdown("#### Copilot Answer:")
            st.info(result['answer'])
            
            with st.expander("🔍 View Retrieved ChromaDB Context Documents"):
                for idx, doc in enumerate(result['retrieved_context']):
                    st.markdown(f"**Retrieved Chunk {idx+1}**:")
                    st.code(doc)

# ----------------------------------------------------
# TAB 4: FORWARD CASH FORECASTER (MONTE CARLO)
# ----------------------------------------------------
with tab4:
    st.markdown("### Forward Cash Position Forecaster (Monte Carlo Simulation)")
    fc = pipeline_data["forecast_summary"]
    sim_totals = pipeline_data["simulated_totals"]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">INR {fc["p10_pessimistic"]:,.2f} <span class="metric-badge-orange">P10</span></div><div class="metric-lbl-sub">Pessimistic (10th Percentile)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">INR {fc["p50_median"]:,.2f} <span class="metric-badge-blue">P50</span></div><div class="metric-lbl-sub">Median Expected Position</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card-dark"><div class="metric-val-large">INR {fc["p90_optimistic"]:,.2f} <span class="metric-badge-green">P90</span></div><div class="metric-lbl-sub">Optimistic (90th Percentile)</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig_hist = px.histogram(
        x=sim_totals, nbins=60,
        labels={"x": "Simulated 30-Day Forward Cash Position (INR)", "y": "Frequency"},
        title="10,000 Monte Carlo Simulation Cash Flow Distribution",
        color_discrete_sequence=['#3B82F6']
    )
    fig_hist.add_vline(x=fc["p10_pessimistic"], line_dash="dash", line_color="#EF4444", annotation_text="P10 (Pessimistic)")
    fig_hist.add_vline(x=fc["p50_median"], line_dash="solid", line_color="#10B981", annotation_text="P50 (Median)")
    fig_hist.add_vline(x=fc["p90_optimistic"], line_dash="dash", line_color="#60A5FA", annotation_text="P90 (Optimistic)")
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC')
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------
# TAB 5: HONEST EXCEPTIONS & LLM SYSTEM SUMMARY
# ----------------------------------------------------
with tab5:
    st.markdown("### Honest Exception List & Executive Explanation Layer")
    exc_df = pipeline_data["honest_exceptions"]
    st.markdown(f"#### ⚠️ Honest Exception List ({len(exc_df)} Items Flagged)")
    st.dataframe(exc_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Executive Explanation with Mathematical Guardrails")
    
    if st.button("Generate Executive Brief"):
        with st.spinner("Synthesizing executive report..."):
            exp_res = generate_system_explanation(
                pipeline_data["summary"],
                pipeline_data["eval_results"],
                pipeline_data["forecast_summary"],
                exc_df,
                groq_api_key=groq_key_input
            )
            st.markdown(f"**Engine**: `{exp_res['llm_engine']}` | **Guardrails**: `{exp_res['guardrail_status']}`")
            st.markdown(exp_res['explanation'])
    else:
        exp_res = generate_system_explanation(
            pipeline_data["summary"],
            pipeline_data["eval_results"],
            pipeline_data["forecast_summary"],
            exc_df,
            groq_api_key=groq_key_input
        )
        st.markdown(f"**Engine**: `{exp_res['llm_engine']}` | **Guardrails**: `{exp_res['guardrail_status']}`")
        st.markdown(exp_res['explanation'])
