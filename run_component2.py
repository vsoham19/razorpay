import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app')))

from ml_engine import build_ml_dataset, train_and_evaluate_ml_models, predict_confidence_for_unmatched

def main():
    print("=" * 80)
    print(" COMPONENT 2: CONFIDENCE SCORING WITH RANDOM FOREST (STRATIFIED GROUP 5-FOLD CV)")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    bank_csv = os.path.join(data_dir, 'bank_stmt.csv')
    ledger_csv = os.path.join(data_dir, 'settlement_ledger.csv')
    truth_csv = os.path.join(data_dir, 'ground_truth.csv')
    results_c1_csv = os.path.join(data_dir, 'matched_results_component1.csv')
    
    df_bank = pd.read_csv(bank_csv)
    df_ledger = pd.read_csv(ledger_csv)
    df_truth = pd.read_csv(truth_csv)
    df_c1 = pd.read_csv(results_c1_csv)
    
    # 1. Build Feature Dataset with Hard Negatives
    print("\n[1] Extracting features with Hard Negative Near-Miss pairs...")
    X, y, df_meta = build_ml_dataset(df_bank, df_ledger, df_truth)
    print(f" Dataset constructed: {X.shape[0]} candidate pairs ({sum(y==1)} positive, {sum(y==0)} negative)")
    
    # 2. Train & Evaluate Models via Stratified Group 5-Fold CV
    n_splits = 5
    print(f"\n[2] Running {n_splits}-Fold Stratified GROUP Cross-Validation (Grouped by Ledger ID)...")
    rf_model, scaler, eval_results = train_and_evaluate_ml_models(X, y, df_meta, n_splits=n_splits)
    
    lr = eval_results['baseline_lr']
    rf = eval_results['random_forest']
    
    print("\n" + "-" * 80)
    print(f" MODEL CROSS-VALIDATION PERFORMANCE ({n_splits}-FOLD GROUP CV - GROUP LEAKAGE PREVENTED)")
    print("-" * 80)
    print(f"{'Metric':<18} | {'Logistic Regression (Baseline)':<30} | {'Random Forest Classifier':<25}")
    print("-" * 80)
    print(f"{'Precision':<18} | {lr['precision']:.4f} ± {lr['precision_std']:.4f}{'':<17} | {rf['precision']:.4f} ± {rf['precision_std']:.4f}")
    print(f"{'Recall':<18} | {lr['recall']:.4f} ± {lr['recall_std']:.4f}{'':<17} | {rf['recall']:.4f} ± {rf['recall_std']:.4f}")
    print(f"{'F1-Score':<18} | {lr['f1']:.4f} ± {lr['f1_std']:.4f}{'':<17} | {rf['f1']:.4f} ± {rf['f1_std']:.4f}")
    print(f"{'ROC-AUC':<18} | {lr['roc_auc']:.4f} ± {lr['roc_auc_std']:.4f}{'':<17} | {rf['roc_auc']:.4f} ± {rf['roc_auc_std']:.4f}")
    print("-" * 80)
    
    print("\n" + "-" * 80)
    print(f" OUT-OF-FOLD (OOF) CONFUSION MATRICES (TOTAL CANDIDATES: {eval_results['total_dataset_size']})")
    print("-" * 80)
    lr_cm = lr['confusion_matrix']
    rf_cm = rf['confusion_matrix']
    
    print("LOGISTIC REGRESSION OOF CONFUSION MATRIX:")
    print(f"  True Negatives (TN) : {lr_cm['true_negatives']:<5} | False Positives (FP): {lr_cm['false_positives']}")
    print(f"  False Negatives (FN): {lr_cm['false_negatives']:<5} | True Positives (TP) : {lr_cm['true_positives']}")
    
    print("\nRANDOM FOREST OOF CONFUSION MATRIX:")
    print(f"  True Negatives (TN) : {rf_cm['true_negatives']:<5} | False Positives (FP): {rf_cm['false_positives']}")
    print(f"  False Negatives (FN): {rf_cm['false_negatives']:<5} | True Positives (TP) : {rf_cm['true_positives']}")
    
    print("\n[3] Random Forest Feature Importances:")
    for feat, imp in eval_results['feature_importances'].items():
        print(f"  - {feat:<22}: {imp:.4f}")
        
    cs = eval_results.get('case_study_example')
    if cs:
        print("\n" + "=" * 80)
        print(" CASE STUDY: LOGISTIC REGRESSION FAILURE VS RANDOM FOREST SUCCESS")
        print("=" * 80)
        print(f"  Ledger Txn ID:    {cs['ledger_txn_id']} | Merchant Order: {cs['order_id']}")
        print(f"  Bank Narration:   {cs['bank_narration']}")
        print(f"  Category:         {cs['category']}")
        print(f"  Ground Truth:     {'MATCH (1)' if cs['ground_truth_label'] == 1 else 'NON-MATCH (0)'}")
        print(f"  Features Extracted:")
        for k, v in cs['features'].items():
            print(f"    * {k:<22}: {v}")
        print(f"  Predictions:")
        print(f"    * Logistic Regression OOF: Pred={cs['logistic_regression_pred']} (Prob={cs['logistic_regression_prob']}) -> {'CORRECT' if cs['logistic_regression_pred'] == cs['ground_truth_label'] else 'WRONG (MISCLASSIFIED)'}")
        print(f"    * Random Forest OOF      : Pred={cs['random_forest_pred']} (Prob={cs['random_forest_prob']}) -> {'CORRECT' if cs['random_forest_pred'] == cs['ground_truth_label'] else 'WRONG'}")
        print("=" * 80)

    # 3. Predict Confidence for Unmatched Records
    print("\n[4] Scoring ML Confidence on TIER_4_UNMATCHED records from Component 1...")
    unmatched_df = df_c1[df_c1['match_tier'] == 'TIER_4_UNMATCHED']
    scored_unmatched = predict_confidence_for_unmatched(rf_model, unmatched_df)
    
    ml_resolved = scored_unmatched[scored_unmatched['ml_prediction'] == 'RESOLVED_BY_ML']
    honest_exceptions = scored_unmatched[scored_unmatched['ml_prediction'] == 'HONEST_EXCEPTION']
    
    print(f" Total Tier 4 Unmatched:  {len(unmatched_df)}")
    print(f" Resolved by ML (Conf >= 0.70): {len(ml_resolved)}")
    print(f" Flagged for Honest Exception List (Conf < 0.70): {len(honest_exceptions)}")
    
    out_scored = os.path.join(data_dir, 'ml_scored_exceptions.csv')
    scored_unmatched.to_csv(out_scored, index=False)
    print(f"\n Saved ML Scored Exceptions -> {out_scored}")
    
if __name__ == "__main__":
    main()
