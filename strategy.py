#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Use weekly pivot levels as trend filter and daily Donchian(20) breakout with volume confirmation. 
Long when price breaks above 20-day high with price above weekly pivot (bullish bias). 
Short when price breaks below 20-day low with price below weekly pivot (bearish bias). 
Weekly pivot provides multi-day trend context to avoid counter-trend trades. 
Volume spike confirms breakout strength. Targets ~15 trades/year on 1d to minimize fee drag.
Works in bull via upside breakouts and bear via downside breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: P = (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to daily timeframe (use previous week's pivot)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        pivot = weekly_pivot_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with bullish bias and volume spike
            if close[i] > upper and close[i] > pivot and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below 20-day low with bearish bias and volume spike
            elif close[i] < lower and close[i] < pivot and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below 20-day low or crosses below weekly pivot
            if close[i] < lower or close[i] < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above 20-day high or crosses above weekly pivot
            if close[i] > upper or close[i] > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "1d"
leverage = 1.0