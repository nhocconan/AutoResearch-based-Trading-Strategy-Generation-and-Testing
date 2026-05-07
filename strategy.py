#!/usr/bin/env python3
"""
1h_Funding_Rate_Contrarian_4hTrend_Filter
Hypothesis: Funding rate extremes indicate short-term sentiment extremes. 
Contrarian entries (short when funding > 0.03%, long when funding < -0.03%) 
work when aligned with 4h trend (EMA50). 
Funding mean-reverts quickly, providing edge in ranging/volatile markets.
4h trend filter ensures we don't fight the medium-term direction.
Session filter (08-20 UTC) reduces noise. Target 15-30 trades/year.
"""
name = "1h_Funding_Rate_Contrarian_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (8h intervals)
    funding_path = "/mnt/raid0_ssd_nvme/data/processed/funding/binance_funding_rate_8h.parquet"
    try:
        funding_df = pd.read_parquet(funding_path)
        funding_series = funding_df.set_index('timestamp')['funding_rate']
    except Exception:
        funding_series = pd.Series(index=prices.index, data=0.0)
    
    # Align funding rate to 1h (forward fill from 8h data)
    funding_aligned = pd.Series(index=prices.index, data=0.0, dtype=float)
    funding_ts = funding_series.index
    funding_vals = funding_series.values
    if len(funding_ts) > 0:
        funding_aligned.values[:] = np.interp(
            prices.index.view('int64'),
            funding_ts.view('int64'),
            funding_vals,
            left=funding_vals[0] if len(funding_vals) > 0 else 0.0,
            right=funding_vals[-1] if len(funding_vals) > 0 else 0.0
        )
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        funding = funding_aligned.iloc[i] if hasattr(funding_aligned, 'iloc') else funding_aligned[i]
        
        if position == 0:
            # Long: funding extremely negative (< -0.03%) AND price above 4h EMA50 (uptrend)
            if funding < -0.0003 and prices['close'].iloc[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: funding extremely positive (> 0.03%) AND price below 4h EMA50 (downtrend)
            elif funding > 0.0003 and prices['close'].iloc[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: funding returns to neutral OR price crosses below EMA50
            if funding > -0.0001 or prices['close'].iloc[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: funding returns to neutral OR price crosses above EMA50
            if funding < 0.0001 or prices['close'].iloc[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals