import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_clear=120, num_ambiguous=50, num_hard_negatives=40, num_unmatched_ledger=15, num_unmatched_bank=15, seed=42):
    """
    Generates realistic paired synthetic data for financial reconciliation benchmarking.
    
    Categories generated:
    1. Clear Matches (~50%): Exact/near-exact date, exact net amount, exact reference ID in narration.
    2. Near-Miss / Ambiguous Positive Matches (~20%): Fee variations, narration typos, settlement lag, partial adjustments.
    3. Hard Negative Near-Misses (~20%): Same amount different order, canceled order retries, duplicate payouts, expired lag.
    4. Clean Non-Matches (~10%): Ledger orphan records (webhook failed) and Bank orphan records (unmatched credits).
    """
    random.seed(seed)
    
    base_date = datetime(2026, 8, 1)
    
    bank_records = []
    ledger_records = []
    ground_truth = []
    
    txn_counter = 1000
    
    # 1. CLEAR MATCHES (~50%)
    for i in range(num_clear):
        txn_counter += 1
        order_id = f"ORD_202608{random.randint(10, 25):02d}_{txn_counter}"
        gross_amt = round(random.uniform(500.0, 15000.0), 2)
        fee_rate = round(random.choice([0.015, 0.02, 0.025]), 4)
        fee = round(gross_amt * fee_rate, 2)
        net_amt = round(gross_amt - fee, 2)
        
        tx_date = base_date + timedelta(days=random.randint(0, 15))
        settlement_delay = random.choice([0, 1, 2])
        bank_date = tx_date + timedelta(days=settlement_delay)
        
        ledger_id = f"LEDGER_{txn_counter}"
        bank_id = f"BANK_{txn_counter + 8000}"
        
        narration_prefix = random.choice(["CMS/RAZORPAY/", "NEFT-RAZORPAY-SETTL/", "UPI/RAZORPAY/PAY/"])
        bank_narration = f"{narration_prefix}{order_id}/UTIB{random.randint(10000, 99999)}"
        
        ledger_records.append({
            "ledger_txn_id": ledger_id,
            "settlement_date": tx_date.strftime("%Y-%m-%d"),
            "merchant_order_id": order_id,
            "gross_amount": gross_amt,
            "fee_deducted": fee,
            "net_amount": net_amt,
            "currency": "INR",
            "status": "SETTLED"
        })
        
        bank_records.append({
            "bank_txn_id": bank_id,
            "bank_date": bank_date.strftime("%Y-%m-%d"),
            "bank_amount": net_amt,
            "bank_narration": bank_narration,
            "currency": "INR"
        })
        
        ground_truth.append({
            "ledger_txn_id": ledger_id,
            "bank_txn_id": bank_id,
            "category": "CLEAR_MATCH",
            "expected_match": True
        })
        
    # 2. NEAR-MISS / AMBIGUOUS POSITIVE MATCHES (~20%)
    for i in range(num_ambiguous):
        txn_counter += 1
        order_id = f"ORD_202608{random.randint(10, 25):02d}_{txn_counter}"
        gross_amt = round(random.uniform(1000.0, 20000.0), 2)
        fee = round(gross_amt * 0.02, 2)
        net_amt = round(gross_amt - fee, 2)
        
        tx_date = base_date + timedelta(days=random.randint(0, 15))
        ledger_id = f"LEDGER_{txn_counter}"
        bank_id = f"BANK_{txn_counter + 8000}"
        
        case_type = random.choice(["FEE_VARIATION", "NARRATION_TYPO", "SETTLEMENT_LAG", "PARTIAL_AMOUNT_DELTA"])
        
        if case_type == "FEE_VARIATION":
            extra_fee = round(gross_amt * random.uniform(0.005, 0.015), 2)
            bank_amt = round(net_amt - extra_fee, 2)
            bank_date = tx_date + timedelta(days=random.randint(1, 3))
            bank_narration = f"RAZORPAY/{order_id}/NET_PAYOUT"
            
        elif case_type == "NARRATION_TYPO":
            typo_order_id = order_id[:-2] + "XX"
            bank_amt = net_amt
            bank_date = tx_date + timedelta(days=random.randint(1, 2))
            bank_narration = f"RAZORPAY_PAYOUT_{typo_order_id}"
            
        elif case_type == "SETTLEMENT_LAG":
            bank_amt = net_amt
            bank_date = tx_date + timedelta(days=random.randint(5, 8))
            bank_narration = f"NEFT-RAZORPAY-{order_id}"
            
        else: # PARTIAL_AMOUNT_DELTA
            adjustment = round(random.uniform(150.0, 500.0), 2)
            bank_amt = max(10.0, round(net_amt - adjustment, 2))
            bank_date = tx_date + timedelta(days=random.randint(2, 4))
            bank_narration = f"RAZORPAY_ADJ_{order_id}"
            
        ledger_records.append({
            "ledger_txn_id": ledger_id,
            "settlement_date": tx_date.strftime("%Y-%m-%d"),
            "merchant_order_id": order_id,
            "gross_amount": gross_amt,
            "fee_deducted": fee,
            "net_amount": net_amt,
            "currency": "INR",
            "status": "SETTLED"
        })
        
        bank_records.append({
            "bank_txn_id": bank_id,
            "bank_date": bank_date.strftime("%Y-%m-%d"),
            "bank_amount": bank_amt,
            "bank_narration": bank_narration,
            "currency": "INR"
        })
        
        ground_truth.append({
            "ledger_txn_id": ledger_id,
            "bank_txn_id": bank_id,
            "category": f"AMBIGUOUS_POS_{case_type}",
            "expected_match": True
        })

    # 3. HARD NEGATIVES / NEAR-MISS NON-MATCHES (~20%)
    # Genuine non-matches that trick simple linear rules!
    for i in range(num_hard_negatives):
        txn_counter += 1
        order_id_ledger = f"ORD_202608{random.randint(10, 25):02d}_{txn_counter}"
        order_id_bank = f"ORD_202608{random.randint(10, 25):02d}_{txn_counter + 5000}"
        
        gross_amt = round(random.uniform(2000.0, 15000.0), 2)
        fee = round(gross_amt * 0.02, 2)
        net_amt = round(gross_amt - fee, 2)
        
        tx_date = base_date + timedelta(days=random.randint(0, 15))
        ledger_id = f"LEDGER_{txn_counter}"
        bank_id = f"BANK_{txn_counter + 8000}"
        
        hard_type = random.choice([
            "SAME_AMOUNT_DIFFERENT_ORDER",
            "SIMILAR_REF_DIFFERENT_AMOUNT",
            "CANCELED_ORDER_RETRY",
            "EXPIRED_PAYOUT_LAG"
        ])
        
        if hard_type == "SAME_AMOUNT_DIFFERENT_ORDER":
            # EXACT same net amount, close date (1 day gap), but completely different order ID!
            # High amount ratio (1.0), low date gap (1 day), but exact_ref = 0 and ref_sim = 0.35.
            bank_amt = net_amt
            bank_date = tx_date + timedelta(days=1)
            bank_narration = f"CMS/RAZORPAY/{order_id_bank}/UTIB7788"
            
        elif hard_type == "SIMILAR_REF_DIFFERENT_AMOUNT":
            # High reference similarity (~0.85), but amount differs by 8% (outside fee tolerance!)
            # High ref_sim (0.85), but amount ratio is 0.92 and fee_pct_diff is 0.08.
            bank_amt = round(net_amt * random.uniform(0.85, 0.91), 2)
            bank_date = tx_date + timedelta(days=random.randint(1, 3))
            typo_order = order_id_ledger[:-2] + "88"
            bank_narration = f"RAZORPAY_PAYOUT_{typo_order}"
            
        elif hard_type == "CANCELED_ORDER_RETRY":
            # Exact order ID string present in bank narration, but date gap is 9 days and amount is 15% less (chargeback penalty)
            # exact_ref = 1.0, but date_gap = 9 days and fee_pct = 15%.
            bank_amt = round(net_amt * 0.85, 2)
            bank_date = tx_date + timedelta(days=9)
            bank_narration = f"RAZORPAY_ADJ_{order_id_ledger}_REV"
            
        else: # EXPIRED_PAYOUT_LAG
            # Date gap is 12 days, amount differs by ₹350, ref similarity is 0.75.
            bank_amt = round(net_amt + 350.0, 2)
            bank_date = tx_date + timedelta(days=12)
            bank_narration = f"RAZORPAY/OLD_EXP_{order_id_ledger[:-3]}"
            
        ledger_records.append({
            "ledger_txn_id": ledger_id,
            "settlement_date": tx_date.strftime("%Y-%m-%d"),
            "merchant_order_id": order_id_ledger,
            "gross_amount": gross_amt,
            "fee_deducted": fee,
            "net_amount": net_amt,
            "currency": "INR",
            "status": "UNMATCHED_DISPUTED"
        })
        
        bank_records.append({
            "bank_txn_id": bank_id,
            "bank_date": bank_date.strftime("%Y-%m-%d"),
            "bank_amount": bank_amt,
            "bank_narration": bank_narration,
            "currency": "INR"
        })
        
        ground_truth.append({
            "ledger_txn_id": ledger_id,
            "bank_txn_id": bank_id,
            "category": f"HARD_NEG_{hard_type}",
            "expected_match": False  # GENUINE HARD NEGATIVE!
        })

    # 4. CLEAN NON-MATCHES - LEDGER ORPHANS (~5%)
    for i in range(num_unmatched_ledger):
        txn_counter += 1
        order_id = f"ORD_202608{random.randint(10, 25):02d}_{txn_counter}"
        gross_amt = round(random.uniform(500.0, 8000.0), 2)
        fee = round(gross_amt * 0.02, 2)
        net_amt = round(gross_amt - fee, 2)
        tx_date = base_date + timedelta(days=random.randint(0, 15))
        ledger_id = f"LEDGER_{txn_counter}"
        
        ledger_records.append({
            "ledger_txn_id": ledger_id,
            "settlement_date": tx_date.strftime("%Y-%m-%d"),
            "merchant_order_id": order_id,
            "gross_amount": gross_amt,
            "fee_deducted": fee,
            "net_amount": net_amt,
            "currency": "INR",
            "status": "PENDING_BANK_CONFIRMATION"
        })
        
        ground_truth.append({
            "ledger_txn_id": ledger_id,
            "bank_txn_id": None,
            "category": "LEDGER_ORPHAN",
            "expected_match": False
        })
        
    # 5. CLEAN NON-MATCHES - BANK ORPHANS (~5%)
    for i in range(num_unmatched_bank):
        txn_counter += 1
        tx_date = base_date + timedelta(days=random.randint(0, 15))
        bank_id = f"BANK_{txn_counter + 8000}"
        bank_amt = round(random.uniform(1000.0, 25000.0), 2)
        
        bank_records.append({
            "bank_txn_id": bank_id,
            "bank_date": tx_date.strftime("%Y-%m-%d"),
            "bank_amount": bank_amt,
            "bank_narration": f"DIRECT_CREDIT_UNRECORDED_REF_{random.randint(100000, 999999)}",
            "currency": "INR"
        })
        
        ground_truth.append({
            "ledger_txn_id": None,
            "bank_txn_id": bank_id,
            "category": "BANK_ORPHAN",
            "expected_match": False
        })
        
    random.shuffle(bank_records)
    random.shuffle(ledger_records)
    
    df_bank = pd.DataFrame(bank_records)
    df_ledger = pd.DataFrame(ledger_records)
    df_truth = pd.DataFrame(ground_truth)
    
    return df_bank, df_ledger, df_truth

if __name__ == "__main__":
    df_bank, df_ledger, df_truth = generate_synthetic_data()
    print(f"Generated {len(df_bank)} bank records and {len(df_ledger)} ledger records.")
