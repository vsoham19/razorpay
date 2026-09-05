import warnings
warnings.filterwarnings('ignore')
import re
import difflib
import pandas as pd
from datetime import datetime

def compute_string_similarity(str1: str, str2: str) -> float:
    """Calculates SequenceMatcher similarity ratio between two reference strings."""
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.upper(), str2.upper()).ratio()

def extract_order_id_from_narration(narration: str) -> str:
    """Extracts ORD_... pattern if present in bank narration."""
    match = re.search(r'ORD_\d{8}_\d+', str(narration))
    return match.group(0) if match else ""

def run_rule_matching(df_bank: pd.DataFrame, df_ledger: pd.DataFrame):
    """
    Executes tiered rule-based reconciliation matching between bank statement and settlement ledger.
    
    Returns:
    - matched_df: DataFrame with matched pairs, tier classification, rule confidence, and feature deltas.
    - summary: Dictionary summarizing match counts and rates.
    """
    matches = []
    
    # Create working copies with converted dates
    bank_df = df_bank.copy()
    ledger_df = df_ledger.copy()
    
    bank_df['bank_dt'] = pd.to_datetime(bank_df['bank_date'])
    ledger_df['settlement_dt'] = pd.to_datetime(ledger_df['settlement_date'])
    
    # Track assigned bank transactions to avoid double matching
    matched_bank_ids = set()
    
    for _, ledger_row in ledger_df.iterrows():
        ledger_id = ledger_row['ledger_txn_id']
        order_id = str(ledger_row['merchant_order_id'])
        gross_amt = float(ledger_row['gross_amount'])
        net_amt = float(ledger_row['net_amount'])
        settle_dt = ledger_row['settlement_dt']
        
        best_match = None
        
        for _, bank_row in bank_df.iterrows():
            bank_id = bank_row['bank_txn_id']
            if bank_id in matched_bank_ids:
                continue
                
            bank_amt = float(bank_row['bank_amount'])
            bank_narration = str(bank_row['bank_narration'])
            bank_dt = bank_row['bank_dt']
            
            # Compute deltas
            amt_diff = abs(bank_amt - net_amt)
            date_gap = abs((bank_dt - settle_dt).days)
            extracted_order = extract_order_id_from_narration(bank_narration)
            
            exact_ref = (order_id in bank_narration) or (extracted_order == order_id)
            fuzzy_ref_score = compute_string_similarity(order_id, bank_narration)
            
            # TIER 1: EXACT MATCH
            if exact_ref and amt_diff < 0.01 and date_gap <= 3:
                best_match = {
                    "ledger_txn_id": ledger_id,
                    "bank_txn_id": bank_id,
                    "merchant_order_id": order_id,
                    "bank_narration": bank_narration,
                    "net_amount": net_amt,
                    "bank_amount": bank_amt,
                    "amount_diff": round(amt_diff, 2),
                    "date_gap_days": date_gap,
                    "ref_similarity": 1.0,
                    "match_tier": "TIER_1_EXACT",
                    "confidence_score": 1.00,
                    "status": "MATCHED"
                }
                matched_bank_ids.add(bank_id)
                break
                
            # TIER 2: FEE/MDR ADJUSTED MATCH
            elif exact_ref and amt_diff <= (gross_amt * 0.035) and date_gap <= 7:
                best_match = {
                    "ledger_txn_id": ledger_id,
                    "bank_txn_id": bank_id,
                    "merchant_order_id": order_id,
                    "bank_narration": bank_narration,
                    "net_amount": net_amt,
                    "bank_amount": bank_amt,
                    "amount_diff": round(amt_diff, 2),
                    "date_gap_days": date_gap,
                    "ref_similarity": 1.0,
                    "match_tier": "TIER_2_FEE_ADJUSTED",
                    "confidence_score": 0.90,
                    "status": "MATCHED"
                }
                matched_bank_ids.add(bank_id)
                break
                
            # TIER 3: FUZZY REFERENCE MATCH
            elif fuzzy_ref_score >= 0.80 and amt_diff < 0.01 and date_gap <= 5:
                best_match = {
                    "ledger_txn_id": ledger_id,
                    "bank_txn_id": bank_id,
                    "merchant_order_id": order_id,
                    "bank_narration": bank_narration,
                    "net_amount": net_amt,
                    "bank_amount": bank_amt,
                    "amount_diff": round(amt_diff, 2),
                    "date_gap_days": date_gap,
                    "ref_similarity": round(fuzzy_ref_score, 4),
                    "match_tier": "TIER_3_FUZZY_REF",
                    "confidence_score": 0.80,
                    "status": "MATCHED"
                }
                matched_bank_ids.add(bank_id)
                break
        
        # TIER 4: UNMATCHED / AMBIGUOUS (FLAGGED FOR ML / EXCEPTION QUEUE)
        if best_match is None:
            # Find closest candidate for ML features
            candidate = None
            min_score = 99999
            for _, bank_row in bank_df.iterrows():
                if bank_row['bank_txn_id'] in matched_bank_ids:
                    continue
                b_amt = float(bank_row['bank_amount'])
                b_dt = bank_row['bank_dt']
                b_nar = str(bank_row['bank_narration'])
                
                a_diff = abs(b_amt - net_amt)
                d_gap = abs((b_dt - settle_dt).days)
                f_sim = compute_string_similarity(order_id, b_nar)
                
                # Combined distance score (lower is closer)
                score = (a_diff * 0.1) + (d_gap * 2) + ((1 - f_sim) * 50)
                if score < min_score:
                    min_score = score
                    candidate = {
                        "bank_txn_id": bank_row['bank_txn_id'],
                        "bank_narration": b_nar,
                        "bank_amount": b_amt,
                        "amount_diff": round(a_diff, 2),
                        "date_gap_days": d_gap,
                        "ref_similarity": round(f_sim, 4)
                    }
                    
            best_match = {
                "ledger_txn_id": ledger_id,
                "bank_txn_id": candidate["bank_txn_id"] if candidate else None,
                "merchant_order_id": order_id,
                "bank_narration": candidate["bank_narration"] if candidate else "",
                "net_amount": net_amt,
                "bank_amount": candidate["bank_amount"] if candidate else 0.0,
                "amount_diff": candidate["amount_diff"] if candidate else 9999.0,
                "date_gap_days": candidate["date_gap_days"] if candidate else 999,
                "ref_similarity": candidate["ref_similarity"] if candidate else 0.0,
                "match_tier": "TIER_4_UNMATCHED",
                "confidence_score": 0.00,
                "status": "UNMATCHED_FLAGGED"
            }
            
        matches.append(best_match)
        
    matched_df = pd.DataFrame(matches)
    
    tier_counts = matched_df['match_tier'].value_counts().to_dict()
    total_ledger = len(ledger_df)
    total_matched = len(matched_df[matched_df['status'] == "MATCHED"])
    
    summary = {
        "total_ledger_records": total_ledger,
        "total_bank_records": len(bank_df),
        "rule_matched_count": total_matched,
        "rule_match_rate": round(total_matched / total_ledger * 100, 2) if total_ledger > 0 else 0.0,
        "tier_breakdown": tier_counts
    }
    
    return matched_df, summary
