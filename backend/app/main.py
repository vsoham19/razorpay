import os
import sys
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'), override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_generator import generate_synthetic_data
from rule_engine import run_rule_matching
from ml_engine import build_ml_dataset, train_and_evaluate_ml_models, predict_confidence_for_unmatched
from rag_engine import index_reconciliation_records, answer_settlement_query
from forecaster import run_monte_carlo_cash_forecast
from explainer import build_honest_exception_list, generate_system_explanation

app = FastAPI(
    title="Settlement Reconciliation & Forecasting Copilot API",
    description="Enterprise AI Finance Controller API for Razorpay Buildathon",
    version="1.0.0"
)

# Global in-memory cache of current system state
STATE = {
    "df_bank": None,
    "df_ledger": None,
    "df_truth": None,
    "df_c1": None,
    "df_ml": None,
    "rf_model": None,
    "eval_results": None,
    "forecast_summary": None,
    "honest_exceptions": None
}

class QueryRequest(BaseModel):
    query: str
    groq_api_key: str = None

class ForecastRequest(BaseModel):
    forecast_days: int = 30
    num_simulations: int = 10000

@app.on_event("startup")
def startup_event():
    # Initialize pipeline with synthetic dataset on startup
    df_bank, df_ledger, df_truth = generate_synthetic_data()
    STATE["df_bank"] = df_bank
    STATE["df_ledger"] = df_ledger
    STATE["df_truth"] = df_truth
    
    # Run C1
    matched_df, summary = run_rule_matching(df_bank, df_ledger)
    STATE["df_c1"] = matched_df
    
    # Run C2
    X, y, df_meta = build_ml_dataset(df_bank, df_ledger, df_truth)
    rf_model, scaler, eval_results = train_and_evaluate_ml_models(X, y, df_meta)
    STATE["rf_model"] = rf_model
    STATE["eval_results"] = eval_results
    
    unmatched_df = matched_df[matched_df['match_tier'] == 'TIER_4_UNMATCHED']
    scored_unmatched = predict_confidence_for_unmatched(rf_model, unmatched_df)
    STATE["df_ml"] = scored_unmatched
    
    # Index records into ChromaDB vector repository for RAG
    index_reconciliation_records(matched_df, scored_unmatched)
    
    # Run C4 (lightweight fast forecast on boot)
    fc_summary, _ = run_monte_carlo_cash_forecast(df_ledger, num_simulations=500)
    STATE["forecast_summary"] = fc_summary
    
    # Run C5
    exc_df = build_honest_exception_list(matched_df, scored_unmatched, fc_summary)
    STATE["honest_exceptions"] = exc_df

from fastapi.responses import FileResponse

@app.get("/")
def read_root():
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'index.html')
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "system": "Settlement Reconciliation & Forecasting Copilot",
        "status": "ONLINE",
        "endpoints": ["/api/reconcile", "/api/ml-eval", "/api/qa", "/api/forecast", "/api/exceptions", "/api/explanation"]
    }

@app.get("/api/reconcile")
def get_reconciliation_results():
    if STATE["df_c1"] is None:
        raise HTTPException(status_code=400, detail="Reconciliation pipeline not initialized.")
    return {
        "records": STATE["df_c1"].to_dict(orient="records"),
        "tier_summary": STATE["df_c1"]['match_tier'].value_counts().to_dict(),
        "total_records": len(STATE["df_c1"])
    }

@app.get("/api/ml-eval")
def get_ml_evaluation():
    if STATE["eval_results"] is None:
        raise HTTPException(status_code=400, detail="ML model not trained.")
    return STATE["eval_results"]

@app.post("/api/qa")
def qa_query(req: QueryRequest):
    res = answer_settlement_query(req.query, groq_api_key=req.groq_api_key)
    return res

@app.post("/api/forecast")
def run_forecast(req: ForecastRequest):
    if STATE["df_ledger"] is None:
        raise HTTPException(status_code=400, detail="No settlement ledger available.")
    fc_summary, _ = run_monte_carlo_cash_forecast(
        STATE["df_ledger"],
        forecast_days=req.forecast_days,
        num_simulations=req.num_simulations
    )
    STATE["forecast_summary"] = fc_summary
    return fc_summary

@app.get("/api/exceptions")
def get_honest_exceptions():
    if STATE["honest_exceptions"] is None:
        raise HTTPException(status_code=400, detail="Honest Exception List not computed.")
    return {
        "count": len(STATE["honest_exceptions"]),
        "exceptions": STATE["honest_exceptions"].to_dict(orient="records")
    }

@app.get("/api/explanation")
def get_executive_explanation(groq_api_key: str = None):
    reconcile_summary = {
        "total_ledger_records": len(STATE["df_c1"]),
        "rule_matched_count": len(STATE["df_c1"][STATE["df_c1"]['status'] == 'MATCHED']),
        "rule_match_rate": round(len(STATE["df_c1"][STATE["df_c1"]['status'] == 'MATCHED']) / len(STATE["df_c1"]) * 100, 2)
    }
    explanation_res = generate_system_explanation(
        reconcile_summary,
        STATE["eval_results"],
        STATE["forecast_summary"],
        STATE["honest_exceptions"],
        groq_api_key=groq_api_key
    )
    return explanation_res
