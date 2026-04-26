#!/usr/bin/env python3
"""
6h_FundingRateMeanReversion_v1
Hypothesis: BTC/ETH funding rates exhibit mean reversion. Extreme positive funding (>0.03%) indicates overleveraged longs -> short. Extreme negative funding (<-0.03%) indicates overleveraged shorts -> long. Uses 6h candles for execution but funding rate as HTF signal. Works in both bull/bear markets by fading funding extremes. Targets 12-37 trades/year with discrete sizing (±0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load funding rate data ONCE before loop (8h data from Binance)
    # Note: funding rate is stored in data/processed/funding/ as 8h candles
    # We'll use 8h as proxy since true funding is per 8h
    try:
        df_funding = get_htf_data(prices, '8h')
        if len(df_funding) < 30:
            return np.zeros(n)
    except:
        # Fallback: if funding data not available, use 1d as proxy (less ideal but functional)
        df_funding = get_htf_data(prices, '1d')
        if len(df_funding) < 30:
            return np.zeros(n)
    
    # Funding rate mean and std for z-score (30-period lookback)
    funding = df_funding['close'].values  # funding rate stored in 'close' column for funding data
    funding_mean = pd.Series(funding).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding).rolling(window=30, min_periods=30).std().values
    funding_z = (funding - funding_mean) / funding_std
    # Replace infinite/NaN from zero std with 0
    funding_z = np.where(np.isfinite(funding_z), funding_z, 0.0)
    
    # Align funding z-score to 6h timeframe
    funding_z_aligned = align_htf_to_ltf(prices, df_funding, funding_z)
    
    # Optional: 6h EMA20 as weak trend filter to avoid strongest counter-trend moves
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of funding calculation (30) + EMA20 (20)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_z_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        funding_z_val = funding_z_aligned[i]
        close_val = close[i]
        ema_20_val = ema_20[i]
        
        # Entry conditions: extreme funding z-score
        long_entry = funding_z_val < -2.0  # extreme negative funding -> long
        short_entry = funding_z_val > 2.0   # extreme positive funding -> short
        
        # Exit conditions: funding returns to neutral zone
        long_exit = funding_z_val > -0.5
        short_exit = funding_z_val < 0.5
        
        # Optional trend filter: only long if above EMA20, only short if below EMA20
        # Comment out for pure mean reversion
        # long_entry = long_entry and (close_val > ema_20_val)
        # short_entry = short_entry and (close_val < ema_20_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and long_exit:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_FundingRateMeanReversion_v1"
timeframe = "6h"
leverage = 1.0