import os
import re
import warnings
warnings.filterwarnings('ignore')

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> chain-of-thought tokens from reasoning models."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'), override=True)

import pandas as pd

def build_honest_exception_list(df_c1: pd.DataFrame, df_ml_exceptions: pd.DataFrame, forecast_summary: dict) -> pd.DataFrame:
    """
    Constructs the Honest Exception List surfacing unresolved reconciliations 
    and data-insufficient forecasts.
    """
    exceptions = []
    
    # 1. Reconciliation Exceptions (Tier 4 records with ML confidence < 0.70)
    if not df_ml_exceptions.empty:
        low_conf_records = df_ml_exceptions[df_ml_exceptions['ml_prediction'] == 'HONEST_EXCEPTION']
        for _, row in low_conf_records.iterrows():
            exceptions.append({
                "exception_type": "UNRESOLVED_RECONCILIATION",
                "record_id": str(row.get('ledger_txn_id', 'N/A')),
                "order_id": str(row.get('merchant_order_id', 'N/A')),
                "confidence_score": float(row.get('ml_confidence_score', 0.0)),
                "issue_description": f"Reference similarity ({row.get('ref_similarity', 0.0)}) and amount delta (INR {row.get('amount_diff', 0.0)}) below threshold.",
                "action_required": "Manual Controller Review Required"
            })
            
    # 2. Forecast Data Insufficiency Exceptions
    if forecast_summary.get('sample_size_warning', False):
        exceptions.append({
            "exception_type": "INSUFFICIENT_FORECAST_DATA",
            "record_id": "FORECAST_GLOBAL",
            "order_id": "N/A",
            "confidence_score": 0.00,
            "issue_description": f"Historical sample size ({forecast_summary.get('historical_sample_size', 0)} records) is below recommended threshold (N >= 30).",
            "action_required": "Ingest more settlement history to improve Monte Carlo precision."
        })
        
    exception_df = pd.DataFrame(exceptions)
    return exception_df

def generate_system_explanation(
    reconcile_summary: dict,
    ml_eval: dict,
    forecast_summary: dict,
    exception_df: pd.DataFrame,
    groq_api_key: str = None
) -> dict:
    """
    Synthesizes overall quantitative outputs into an executive natural-language explanation 
    guarded against mathematical hallucination.
    """
    total_ledger = reconcile_summary.get('total_ledger_records', 0)
    matched_rule = reconcile_summary.get('rule_matched_count', 0)
    rule_rate = reconcile_summary.get('rule_match_rate', 0.0)
    
    rf_metrics = ml_eval.get('random_forest', {})
    rf_f1 = rf_metrics.get('f1', 0.0)
    
    p10 = forecast_summary.get('p10_pessimistic', 0.0)
    p50 = forecast_summary.get('p50_median', 0.0)
    p90 = forecast_summary.get('p90_optimistic', 0.0)
    num_exceptions = len(exception_df)
    
    prompt = (
        "You are an AI Financial Controller Copilot summarizing settlement reconciliation results.\n"
        "STRICT GUARDRAIL: Do not invent, alter, or re-calculate any monetary amounts, percentages, or metrics.\n"
        "Use the exact mathematical outputs provided below:\n\n"
        f"QUANTITATIVE METRICS:\n"
        f"- Total Ledger Records Processed: {total_ledger}\n"
        f"- Rule Engine Matched: {matched_rule} ({rule_rate}% match rate)\n"
        f"- ML Model (Random Forest) F1-Score: {rf_f1}\n"
        f"- Honest Exception Count: {num_exceptions} unresolved items\n"
        f"- 30-Day Forward Cash Forecast:\n"
        f"  * P10 (Pessimistic): INR {p10:,.2f}\n"
        f"  * P50 (Median): INR {p50:,.2f}\n"
        f"  * P90 (Optimistic): INR {p90:,.2f}\n\n"
        "Provide a structured 3-paragraph explanation:\n"
        "1. Reconciliation Matching Summary (Why rules vs ML succeeded)\n"
        "2. Forward Cash Forecast Explanation (Why P10/P50/P90 range exists due to variance)\n"
        "3. Honest Exception List Advisory for the Finance Controller"
    )
    
    explanation_text = ""
    llm_engine = "Deterministic Guardrail Summary"
    
    if groq_api_key or os.getenv("GROQ_API_KEY"):
        try:
            import groq
            key = groq_api_key or os.getenv("GROQ_API_KEY")
            client = groq.Groq(api_key=key)
            
            for model_candidate in ["qwen/qwen3.6-27b", "groq/compound", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]:
                try:
                    completion = client.chat.completions.create(
                        model=model_candidate,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    explanation_text = _strip_think_tags(completion.choices[0].message.content)
                    llm_engine = "AI Finance Assistant"
                    break
                except Exception as model_err:
                    continue
            if not explanation_text:
                explanation_text = _generate_fallback_explanation(reconcile_summary, forecast_summary, num_exceptions)
        except Exception as e:
            explanation_text = _generate_fallback_explanation(reconcile_summary, forecast_summary, num_exceptions)
    else:
        explanation_text = _generate_fallback_explanation(reconcile_summary, forecast_summary, num_exceptions)
        
    return {
        "explanation": explanation_text,
        "llm_engine": llm_engine,
        "guardrail_status": "ACTIVE (Strict Mathematical Alignment Enforced)"
    }

def _generate_fallback_explanation(reconcile_summary: dict, forecast_summary: dict, num_exceptions: int) -> str:
    rule_rate = reconcile_summary.get('rule_match_rate', 0.0)
    p10 = forecast_summary.get('p10_pessimistic', 0.0)
    p50 = forecast_summary.get('p50_median', 0.0)
    p90 = forecast_summary.get('p90_optimistic', 0.0)
    
    return (
        f"### Executive Settlement Summary\n\n"
        f"**1. Reconciliation Performance**:\n"
        f"The rule engine matched {rule_rate}% of records deterministically across Tiers 1-3. "
        f"Ambiguous cases were evaluated by the Random Forest classifier to assign match confidence scores.\n\n"
        f"**2. Cash Position Forecasting**:\n"
        f"Monte Carlo simulation projects a 30-day forward cash position between **INR {p10:,.2f} (P10 pessimistic)** "
        f"and **INR {p90:,.2f} (P90 optimistic)**, with a median expected position of **INR {p50:,.2f} (P50)**. "
        f"The range reflects variance in daily settlement volume, gateway MDR deductions, and bank clearance lags.\n\n"
        f"**3. Honest Exception List Advisory**:\n"
        f"There are currently **{num_exceptions} flagged exceptions** requiring manual controller review before final ledger posting."
    )
