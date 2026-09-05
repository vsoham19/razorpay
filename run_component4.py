import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app')))

from forecaster import run_monte_carlo_cash_forecast

def main():
    print("=" * 60)
    print(" COMPONENT 4: FORWARD CASH FORECASTER (MONTE CARLO)")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ledger_csv = os.path.join(data_dir, 'settlement_ledger.csv')
    df_ledger = pd.read_csv(ledger_csv) if os.path.exists(ledger_csv) else pd.DataFrame()
    
    print(f"\n[1] Ingested {len(df_ledger)} historical settlement records for variance modeling...")
    
    print("\n[2] Running 10,000 Monte Carlo simulations over 30-day horizon...")
    summary, simulated_totals = run_monte_carlo_cash_forecast(
        df_ledger, forecast_days=30, num_simulations=10000, seed=42
    )
    
    print("\n" + "-" * 55)
    print(" 30-DAY FORWARD CASH POSITION DISTRIBUTION (INR)")
    print("-" * 55)
    print(f" P10 (Pessimistic - 10% worst-case) : INR {summary['p10_pessimistic']:,.2f}")
    print(f" P50 (Median Expected Position)     : INR {summary['p50_median']:,.2f}")
    print(f" P90 (Optimistic - 90% percentile)  : INR {summary['p90_optimistic']:,.2f}")
    print("-" * 55)
    print(f" Mean Forecast Position             : INR {summary['mean_forecast']:,.2f}")
    print(f" Forecast Standard Deviation        : INR {summary['std_dev_forecast']:,.2f}")
    print(f" Honest Exception Status            : {summary['honest_exception']}")
    print("-" * 55)
    
if __name__ == "__main__":
    main()
