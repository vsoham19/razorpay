import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def run_monte_carlo_cash_forecast(
    df_ledger: pd.DataFrame,
    forecast_days: int = 30,
    num_simulations: int = 10000,
    seed: int = 42
) -> dict:
    """
    Simulates forward 30-day cash position uncertainty using Monte Carlo simulation.
    
    Returns:
    - percentiles: P10, P50, P90 forecasted net cash positions.
    - simulation_summary: Mean, Std Dev, Min, Max, distribution curve samples.
    - honest_exception: Flagged if historical transaction sample size is too small (< 30 transactions).
    """
    np.random.seed(seed)
    
    num_historical_records = len(df_ledger)
    sample_size_warning = num_historical_records < 30
    
    if num_historical_records > 0:
        daily_amounts = df_ledger['net_amount'].values
        mean_daily_val = float(np.mean(daily_amounts))
        std_daily_val = float(np.std(daily_amounts)) if len(daily_amounts) > 1 else mean_daily_val * 0.2
        
        # Estimate MDR fee deduction percentage variance
        fee_ratios = (df_ledger['fee_deducted'] / df_ledger['gross_amount']).dropna().values
        mean_fee_pct = float(np.mean(fee_ratios)) if len(fee_ratios) > 0 else 0.02
        std_fee_pct = float(np.std(fee_ratios)) if len(fee_ratios) > 1 else 0.005
    else:
        # Default fallback values if empty
        mean_daily_val = 5000.0
        std_daily_val = 1500.0
        mean_fee_pct = 0.02
        std_fee_pct = 0.005
        
    simulated_totals = np.zeros(num_simulations)
    
    for sim in range(num_simulations):
        # Sample daily cash inflows for forecast_days
        daily_inflows = np.random.normal(loc=mean_daily_val, scale=std_daily_val, size=forecast_days)
        daily_inflows = np.maximum(0.0, daily_inflows)  # Non-negative
        
        # Sample daily fee/dispute deductions
        daily_fee_rates = np.random.normal(loc=mean_fee_pct, scale=std_fee_pct, size=forecast_days)
        daily_fee_rates = np.clip(daily_fee_rates, 0.005, 0.08)
        
        # Sample settlement delay lag penalty (e.g. 5% probability of holiday lag reducing immediate liquidity)
        lag_factors = np.random.choice([1.0, 0.95, 0.90, 0.85], size=forecast_days, p=[0.75, 0.15, 0.07, 0.03])
        
        daily_net_settled = daily_inflows * (1.0 - daily_fee_rates) * lag_factors
        simulated_totals[sim] = np.sum(daily_net_settled)
        
    p10 = float(np.percentile(simulated_totals, 10))
    p50 = float(np.percentile(simulated_totals, 50))
    p90 = float(np.percentile(simulated_totals, 90))
    
    summary = {
        "forecast_horizon_days": forecast_days,
        "num_simulations": num_simulations,
        "historical_sample_size": num_historical_records,
        "p10_pessimistic": round(p10, 2),
        "p50_median": round(p50, 2),
        "p90_optimistic": round(p90, 2),
        "mean_forecast": round(float(np.mean(simulated_totals)), 2),
        "std_dev_forecast": round(float(np.std(simulated_totals)), 2),
        "sample_size_warning": sample_size_warning,
        "honest_exception": "INSUFFICIENT_HISTORICAL_DATA" if sample_size_warning else "HEALTHY"
    }
    
    return summary, simulated_totals
