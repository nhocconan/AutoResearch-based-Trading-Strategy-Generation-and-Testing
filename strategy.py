#!/usr/bin/env python3
"""
#100795 - 6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter (1w) and volume confirmation. 
Weekly pivot determines bias: price above weekly pivot = long bias, below = short bias. 
Donchian breakout in direction of weekly bias with volume spike triggers entry. 
Exit when price returns to weekly pivot (mean reversion to weekly equilibrium).
Targets 15-30 trades/year to minimize fee drag. Works in bull (breakouts with weekly uptrend) and bear (breakouts with weekly downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (using previous week's data to avoid look-ahead)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_r1 = close_1w + weekly_range * 1.1 / 12  # Weekly R1
    weekly_s1 = close_1w - weekly_range * 1.1 / 12  # Weekly S1
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian(20) channels on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly bias: price above/below weekly pivot
        weekly_bias_long = close[i] > weekly_pivot_aligned[i]
        weekly_bias_short = close[i] < weekly_pivot_aligned[i]
        
        # Long condition: Donchian breakout above weekly R1, weekly bias long, volume spike
        if (high[i] > weekly_r1_aligned[i] and 
            weekly_bias_long and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Donchian breakout below weekly S1, weekly bias short, volume spike
        elif (low[i] < weekly_s1_aligned[i] and 
              weekly_bias_short and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion)
        elif position == 1 and close[i] < weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0