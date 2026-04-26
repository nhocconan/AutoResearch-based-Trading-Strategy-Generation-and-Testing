#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_v2
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (1w high/low) and volume confirmation.
Only long when price breaks above 6h Donchian high AND weekly pivot high acts as support (price > weekly low),
short when price breaks below 6h Donchian low AND weekly pivot low acts as resistance (price < weekly high).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by combining price structure (Donchian) with weekly pivot levels as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for weekly pivot levels (high/low of previous week)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: Donchian breakout up + price above weekly low (support) + volume spike
        if close[i] > donchian_high[i] and close[i] > weekly_low_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Donchian breakout down + price below weekly high (resistance) + volume spike
        elif close[i] < donchian_low[i] and close[i] < weekly_high_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price retrace to mid-channel or loss of volume confirmation
        elif position == 1 and (close[i] < (donchian_high[i] + donchian_low[i]) / 2 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (donchian_high[i] + donchian_low[i]) / 2 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_v2"
timeframe = "6h"
leverage = 1.0