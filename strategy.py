#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakout on 6h combined with weekly pivot direction (from 1w high/low/close) 
and volume confirmation. Uses weekly pivot to determine longer-term bias: price above weekly pivot 
= bullish bias (favor longs), below = bearish bias (favor shorts). Volume spike avoids false breakouts.
Designed for 15-35 trades per year on 6h timeframe, works in bull via breakouts above weekly pivot,
and in bear via breakdowns below weekly pivot. Weekly pivot provides structural support/resistance
that holds across market regimes.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # We mainly need the pivot level for bias, S1/R1 for context
    
    # Align weekly pivot to 6h timeframe (using previous week's pivot)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Donchian channel (20-period) on 6h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume spike: volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian calculation
    start_idx = period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        pivot_1w_val = pivot_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume spike
            if high[i] > highest_high_val and close[i] > pivot_1w_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot AND volume spike
            elif low[i] < lowest_low_val and close[i] < pivot_1w_val and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR drops below weekly pivot
            if low[i] < lowest_low_val or close[i] < pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR rises above weekly pivot
            if high[i] > highest_high_val or close[i] > pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0