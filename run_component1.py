import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Add backend/app to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app')))

from data_generator import generate_synthetic_data
from rule_engine import run_rule_matching

def main():
    print("=" * 60)
    print(" COMPONENT 1: RECONCILIATION MATCHING ENGINE (RULE-BASED)")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Generate Synthetic Data
    print("\n[1] Generating synthetic transaction datasets...")
    df_bank, df_ledger, df_truth = generate_synthetic_data(
        num_clear=120, 
        num_ambiguous=50, 
        num_unmatched_ledger=15, 
        num_unmatched_bank=15, 
        seed=42
    )
    
    bank_csv = os.path.join(data_dir, 'bank_stmt.csv')
    ledger_csv = os.path.join(data_dir, 'settlement_ledger.csv')
    truth_csv = os.path.join(data_dir, 'ground_truth.csv')
    
    df_bank.to_csv(bank_csv, index=False)
    df_ledger.to_csv(ledger_csv, index=False)
    df_truth.to_csv(truth_csv, index=False)
    print(f" Saved Bank Statement ({len(df_bank)} records) -> {bank_csv}")
    print(f" Saved Settlement Ledger ({len(df_ledger)} records) -> {ledger_csv}")
    
    # 2. Run Tiered Rule Matching
    print("\n[2] Running Tiered Rule Matching Engine...")
    matched_df, summary = run_rule_matching(df_bank, df_ledger)
    
    results_csv = os.path.join(data_dir, 'matched_results_component1.csv')
    matched_df.to_csv(results_csv, index=False)
    print(f" Saved Matched Results -> {results_csv}")
    
    # 3. Print Match Summary
    print("\n" + "-" * 40)
    print(" RULE ENGINE MATCH SUMMARY")
    print("-" * 40)
    print(f"Total Ledger Records: {summary['total_ledger_records']}")
    print(f"Total Bank Records:   {summary['total_bank_records']}")
    print(f"Rule Matched Count:   {summary['rule_matched_count']}")
    print(f"Rule Match Rate:      {summary['rule_match_rate']}%")
    print("\nTier Breakdown:")
    for tier, count in summary['tier_breakdown'].items():
        print(f"  - {tier:22s}: {count:3d} records")
        
    # 4. Print Sample Records for Inspection
    print("\n" + "-" * 60)
    print(" SAMPLE SYNTHETIC MATCHES BY TIER")
    print("-" * 60)
    
    tiers = ["TIER_1_EXACT", "TIER_2_FEE_ADJUSTED", "TIER_3_FUZZY_REF", "TIER_4_UNMATCHED"]
    for t in tiers:
        sample = matched_df[matched_df['match_tier'] == t].head(1)
        if not sample.empty:
            row = sample.iloc[0]
            print(f"\n--- Tier: {t} (Confidence: {row['confidence_score']}) ---")
            print(f"  Ledger ID:        {row['ledger_txn_id']} | Merchant Order: {row['merchant_order_id']}")
            print(f"  Bank ID:          {row['bank_txn_id']} | Bank Narration: {row['bank_narration']}")
            print(f"  Net Amt (Ledger): INR {row['net_amount']} | Bank Amt: INR {row['bank_amount']} (Diff: INR {row['amount_diff']})")
            print(f"  Date Gap:         {row['date_gap_days']} days | Ref Similarity: {row['ref_similarity']}")
            
if __name__ == "__main__":
    main()
