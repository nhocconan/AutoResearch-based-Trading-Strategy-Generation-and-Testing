#!/usr/bin/env python3
"""
1d_Weekly_Donchian_20_With_Volume_Filter
Hypothesis: Daily Donchian breakout (20-day high/low) with weekly trend filter and volume confirmation. 
In bull markets, price breaks above 20-day high with upward weekly trend; in bear markets, breaks below 20-day low with downward weekly trend. 
Volume filter ensures breakout authenticity. Designed for low frequency (~10-20 trades/year) to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly trend: EMA(34) on weekly close ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Weekly volume average for confirmation ===
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day Donchian, 34-week EMA, 20-week volume average
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current weekly volume
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Volume filter: current volume > 1.5x 20-week average
        vol_filter = vol_1w_current > 1.5 * vol_avg_20_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > Donchian high + weekly EMA rising + volume
            if close[i] > donchian_high[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Donchian low + weekly EMA falling + volume
            elif close[i] < donchian_low[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal or opposite breakout
        elif position == 1:
            if close[i] < donchian_low[i]:  # break below Donchian low = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > donchian_high[i]:  # break above Donchian high = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_20_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0