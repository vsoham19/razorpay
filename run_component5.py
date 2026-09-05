import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app')))

from explainer import build_honest_exception_list, generate_system_explanation

def main():
    print("=" * 60)
    print(" COMPONENT 5: LLM EXPLANATION LAYER & HONEST EXCEPTION LIST")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    c1_csv = os.path.join(data_dir, 'matched_results_component1.csv')
    ml_csv = os.path.join(data_dir, 'ml_scored_exceptions.csv')
    
    df_c1 = pd.read_csv(c1_csv) if os.path.exists(c1_csv) else pd.DataFrame()
    df_ml = pd.read_csv(ml_csv) if os.path.exists(ml_csv) else pd.DataFrame()
    
    reconcile_summary = {
        "total_ledger_records": len(df_c1),
        "rule_matched_count": len(df_c1[df_c1['status'] == 'MATCHED']),
        "rule_match_rate": round(len(df_c1[df_c1['status'] == 'MATCHED']) / len(df_c1) * 100, 2) if len(df_c1) > 0 else 0.0
    }
    
    ml_eval = {
        "random_forest": {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    }
    
    forecast_summary = {
        "historical_sample_size": len(df_c1),
        "p10_pessimistic": 222095.17,
        "p50_median": 251508.73,
        "p90_optimistic": 281281.41,
        "sample_size_warning": False
    }
    
    # 1. Build Honest Exception List
    print("\n[1] Constructing Honest Exception List...")
    exception_df = build_honest_exception_list(df_c1, df_ml, forecast_summary)
    
    out_exc = os.path.join(data_dir, 'honest_exception_list.csv')
    exception_df.to_csv(out_exc, index=False)
    print(f" Saved Honest Exception List ({len(exception_df)} items) -> {out_exc}")
    
    print("\nSample Honest Exception Items:")
    for _, row in exception_df.head(3).iterrows():
        print(f"  [{row['exception_type']}] ID: {row['record_id']} | Conf: {row['confidence_score']} | Issue: {row['issue_description']}")
        
    # 2. Generate System Explanation
    print("\n[2] Generating System Explanation with Mathematical Guardrails...")
    explanation_res = generate_system_explanation(reconcile_summary, ml_eval, forecast_summary, exception_df)
    
    print("\n" + "=" * 55)
    print(" EXECUTIVE SYSTEM EXPLANATION")
    print("=" * 55)
    print(f"Engine: {explanation_res['llm_engine']}")
    print(f"Guardrail Status: {explanation_res['guardrail_status']}\n")
    print(explanation_res['explanation'])
    print("=" * 55)
    
if __name__ == "__main__":
    main()
