import warnings
warnings.filterwarnings('ignore')

import difflib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def compute_similarity(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str(str1).upper(), str(str2).upper()).ratio()

def extract_features_from_pair(ledger_row: pd.Series, bank_row: pd.Series) -> dict:
    """Extracts numerical & string similarity features for ML classification."""
    net_amt = float(ledger_row['net_amount'])
    gross_amt = float(ledger_row['gross_amount'])
    bank_amt = float(bank_row['bank_amount']) if bank_row is not None else 0.0
    
    order_id = str(ledger_row['merchant_order_id'])
    narration = str(bank_row['bank_narration']) if bank_row is not None else ""
    
    settle_dt = pd.to_datetime(ledger_row['settlement_date'])
    bank_dt = pd.to_datetime(bank_row['bank_date']) if bank_row is not None else settle_dt
    
    amount_diff_abs = abs(bank_amt - net_amt)
    amount_ratio = min(bank_amt, net_amt) / max(bank_amt, net_amt) if max(bank_amt, net_amt) > 0 else 0.0
    fee_pct_diff = amount_diff_abs / gross_amt if gross_amt > 0 else 0.0
    date_gap_days = abs((bank_dt - settle_dt).days)
    ref_string_similarity = compute_similarity(order_id, narration)
    exact_ref_match = 1.0 if order_id in narration else 0.0
    
    return {
        "amount_diff_abs": amount_diff_abs,
        "amount_ratio": amount_ratio,
        "fee_pct_diff": fee_pct_diff,
        "date_gap_days": date_gap_days,
        "ref_string_similarity": ref_string_similarity,
        "exact_ref_match": exact_ref_match
    }

def build_ml_dataset(df_bank: pd.DataFrame, df_ledger: pd.DataFrame, df_truth: pd.DataFrame):
    """
    Constructs a balanced machine learning dataset of positive and negative candidate pairs,
    including hard negative near-misses (same amount different order, canceled order retries, fee overlaps).
    """
    feature_rows = []
    labels = []
    metadata = []
    
    bank_dict = df_bank.set_index('bank_txn_id').to_dict('index')
    ledger_dict = df_ledger.set_index('ledger_txn_id').to_dict('index')
    
    # 1. Ground Truth Pairs (Positive samples + Hard Negative near-misses)
    for _, truth_row in df_truth.iterrows():
        ledger_id = truth_row['ledger_txn_id']
        bank_id = truth_row['bank_txn_id']
        is_match = truth_row['expected_match']
        category = truth_row['category']
        
        if ledger_id and bank_id and ledger_id in ledger_dict and bank_id in bank_dict:
            l_row = pd.Series(ledger_dict[ledger_id])
            b_row = pd.Series(bank_dict[bank_id])
            feats = extract_features_from_pair(l_row, b_row)
            feature_rows.append(feats)
            labels.append(1 if is_match else 0)
            metadata.append({
                "ledger_txn_id": ledger_id,
                "bank_txn_id": bank_id,
                "group_id": ledger_id,  # Group identifier for StratifiedGroupKFold
                "order_id": l_row['merchant_order_id'],
                "narration": b_row['bank_narration'],
                "net_amount": l_row['net_amount'],
                "bank_amount": b_row['bank_amount'],
                "category": category
            })
            
    # 2. Hard Negative Candidate Pairs (Random non-matching ledger-bank pairs)
    bank_ids = list(bank_dict.keys())
    for ledger_id, l_info in ledger_dict.items():
        l_row = pd.Series(l_info)
        random_bank_ids = np.random.choice(bank_ids, size=1, replace=False)
        for r_b_id in random_bank_ids:
            b_row = pd.Series(bank_dict[r_b_id])
            matched_truth = df_truth[(df_truth['ledger_txn_id'] == ledger_id) & (df_truth['bank_txn_id'] == r_b_id)]
            if matched_truth.empty or not matched_truth.iloc[0]['expected_match']:
                feats = extract_features_from_pair(l_row, b_row)
                feature_rows.append(feats)
                labels.append(0)
                metadata.append({
                    "ledger_txn_id": ledger_id,
                    "bank_txn_id": r_b_id,
                    "group_id": ledger_id,  # Group identifier to prevent Group-Leakage
                    "order_id": l_row['merchant_order_id'],
                    "narration": b_row['bank_narration'],
                    "net_amount": l_row['net_amount'],
                    "bank_amount": b_row['bank_amount'],
                    "category": "RANDOM_NEGATIVE"
                })
                
    X = pd.DataFrame(feature_rows)
    y = np.array(labels)
    df_meta = pd.DataFrame(metadata)
    return X, y, df_meta

def train_and_evaluate_ml_models(X: pd.DataFrame, y: np.ndarray, df_meta: pd.DataFrame = None, n_splits: int = 5):
    """
    Evaluates Logistic Regression Baseline & Random Forest Classifier using 5-Fold STRATIFIED GROUP Cross-Validation,
    grouping strictly on 'group_id' (ledger_txn_id) to eliminate Group-Leakage between train and validation folds.
    """
    groups = df_meta['group_id'].values if df_meta is not None else np.arange(len(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Trackers for Out-of-Fold (OOF) predictions
    lr_oof_preds = np.zeros(len(y))
    lr_oof_probs = np.zeros(len(y))
    rf_oof_preds = np.zeros(len(y))
    rf_oof_probs = np.zeros(len(y))
    
    # Trackers for fold metrics
    lr_fold_prec, lr_fold_rec, lr_fold_f1, lr_fold_auc = [], [], [], []
    rf_fold_prec, rf_fold_rec, rf_fold_f1, rf_fold_auc = [], [], [], []
    
    scaler = StandardScaler()
    
    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
        y_train_f, y_test_f = y[train_idx], y[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_test_scaled = scaler.transform(X_test_f)
        
        # 1. Logistic Regression Baseline
        lr_fold = LogisticRegression(max_iter=1000, random_state=42)
        lr_fold.fit(X_train_scaled, y_train_f)
        lr_p = lr_fold.predict(X_test_scaled)
        lr_prob = lr_fold.predict_proba(X_test_scaled)[:, 1]
        
        lr_oof_preds[test_idx] = lr_p
        lr_oof_probs[test_idx] = lr_prob
        
        lr_fold_prec.append(precision_score(y_test_f, lr_p, zero_division=0))
        lr_fold_rec.append(recall_score(y_test_f, lr_p, zero_division=0))
        lr_fold_f1.append(f1_score(y_test_f, lr_p, zero_division=0))
        lr_fold_auc.append(roc_auc_score(y_test_f, lr_prob))
        
        # 2. Random Forest Classifier
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_fold.fit(X_train_f, y_train_f)
        rf_p = rf_fold.predict(X_test_f)
        rf_prob = rf_fold.predict_proba(X_test_f)[:, 1]
        
        rf_oof_preds[test_idx] = rf_p
        rf_oof_probs[test_idx] = rf_prob
        
        rf_fold_prec.append(precision_score(y_test_f, rf_p, zero_division=0))
        rf_fold_rec.append(recall_score(y_test_f, rf_p, zero_division=0))
        rf_fold_f1.append(f1_score(y_test_f, rf_p, zero_division=0))
        rf_fold_auc.append(roc_auc_score(y_test_f, rf_prob))

    # Calculate Out-Of-Fold (OOF) Confusion Matrices
    lr_cm = confusion_matrix(y, lr_oof_preds)
    rf_cm = confusion_matrix(y, rf_oof_preds)
    
    lr_metrics = {
        "model": "Logistic Regression (Baseline - Stratified Group 5-Fold CV)",
        "precision": round(float(np.mean(lr_fold_prec)), 4),
        "precision_std": round(float(np.std(lr_fold_prec)), 4),
        "recall": round(float(np.mean(lr_fold_rec)), 4),
        "recall_std": round(float(np.std(lr_fold_rec)), 4),
        "f1": round(float(np.mean(lr_fold_f1)), 4),
        "f1_std": round(float(np.std(lr_fold_f1)), 4),
        "roc_auc": round(float(np.mean(lr_fold_auc)), 4),
        "roc_auc_std": round(float(np.std(lr_fold_auc)), 4),
        "confusion_matrix": {
            "true_negatives": int(lr_cm[0, 0]),
            "false_positives": int(lr_cm[0, 1]),
            "false_negatives": int(lr_cm[1, 0]),
            "true_positives": int(lr_cm[1, 1])
        }
    }
    
    rf_metrics = {
        "model": "Random Forest Classifier (Stratified Group 5-Fold CV)",
        "precision": round(float(np.mean(rf_fold_prec)), 4),
        "precision_std": round(float(np.std(rf_fold_prec)), 4),
        "recall": round(float(np.mean(rf_fold_rec)), 4),
        "recall_std": round(float(np.std(rf_fold_rec)), 4),
        "f1": round(float(np.mean(rf_fold_f1)), 4),
        "f1_std": round(float(np.std(rf_fold_f1)), 4),
        "roc_auc": round(float(np.mean(rf_fold_auc)), 4),
        "roc_auc_std": round(float(np.std(rf_fold_auc)), 4),
        "confusion_matrix": {
            "true_negatives": int(rf_cm[0, 0]),
            "false_positives": int(rf_cm[0, 1]),
            "false_negatives": int(rf_cm[1, 0]),
            "true_positives": int(rf_cm[1, 1])
        }
    }
    
    # Train final full Random Forest model for inference
    final_rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_rf_model.fit(X, y)
    
    feature_importances = dict(zip(X.columns, [round(val, 4) for val in final_rf_model.feature_importances_]))
    
    # Extract concrete case study where Logistic Regression fails OOF and Random Forest succeeds OOF
    case_study = None
    if df_meta is not None and not df_meta.empty:
        discrepant_indices = np.where((lr_oof_preds != y) & (rf_oof_preds == y))[0]
        if len(discrepant_indices) > 0:
            idx = discrepant_indices[0]
            row_meta = df_meta.iloc[idx].to_dict()
            row_feats = X.iloc[idx].to_dict()
            case_study = {
                "ledger_txn_id": row_meta.get("ledger_txn_id", "N/A"),
                "order_id": row_meta.get("order_id", "N/A"),
                "bank_narration": row_meta.get("narration", "N/A"),
                "category": row_meta.get("category", "N/A"),
                "ground_truth_label": int(y[idx]),
                "logistic_regression_pred": int(lr_oof_preds[idx]),
                "logistic_regression_prob": round(float(lr_oof_probs[idx]), 4),
                "random_forest_pred": int(rf_oof_preds[idx]),
                "random_forest_prob": round(float(rf_oof_probs[idx]), 4),
                "features": {k: round(float(v), 4) for k, v in row_feats.items()}
            }
            
    eval_results = {
        "n_splits": n_splits,
        "total_dataset_size": len(y),
        "group_leakage_prevented": True,
        "baseline_lr": lr_metrics,
        "random_forest": rf_metrics,
        "feature_importances": feature_importances,
        "case_study_example": case_study
    }
    
    return final_rf_model, scaler, eval_results

def predict_confidence_for_unmatched(rf_model, df_unmatched: pd.DataFrame):
    """
    Computes ML match confidence scores for TIER_4_UNMATCHED records.
    """
    if df_unmatched.empty:
        return df_unmatched
        
    feature_cols = ["amount_diff_abs", "amount_ratio", "fee_pct_diff", "date_gap_days", "ref_string_similarity", "exact_ref_match"]
    
    unmatched_feats = []
    for _, row in df_unmatched.iterrows():
        bank_amt = float(row['bank_amount'])
        net_amt = float(row['net_amount'])
        amt_diff = float(row['amount_diff'])
        amt_ratio = min(bank_amt, net_amt) / max(bank_amt, net_amt) if max(bank_amt, net_amt) > 0 else 0.0
        fee_pct = amt_diff / (net_amt + 50.0)
        date_gap = int(row['date_gap_days'])
        ref_sim = float(row['ref_similarity'])
        exact_ref = 1.0 if ref_sim >= 0.99 else 0.0
        
        unmatched_feats.append({
            "amount_diff_abs": amt_diff,
            "amount_ratio": amt_ratio,
            "fee_pct_diff": fee_pct,
            "date_gap_days": date_gap,
            "ref_string_similarity": ref_sim,
            "exact_ref_match": exact_ref
        })
        
    X_unmatched = pd.DataFrame(unmatched_feats)[feature_cols]
    probabilities = rf_model.predict_proba(X_unmatched)[:, 1]
    
    scored_df = df_unmatched.copy()
    scored_df['ml_confidence_score'] = [round(p, 4) for p in probabilities]
    scored_df['ml_prediction'] = ["RESOLVED_BY_ML" if p >= 0.70 else "HONEST_EXCEPTION" for p in probabilities]
    
    return scored_df
