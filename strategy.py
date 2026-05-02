#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Weekly pivot provides institutional structure from higher timeframe
# Donchian breakout captures momentum with defined risk
# Volume confirmation (1.5x 20-period average) filters weak breakouts
# Works in bull markets via upward breakouts and bear markets via downward breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Donchian20_1wPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d weekly pivot points (prior completed week's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed weekly bar's high, low, close for pivot
    wh = pd.Series(df_1w['high']).shift(1).values
    wl = pd.Series(df_1w['low']).shift(1).values
    wc = pd.Series(df_1w['close']).shift(1).values
    
    # Weekly pivot point and support/resistance levels
    pivot = (wh + wl + wc) / 3.0
    r1 = 2 * pivot - wl
    s1 = 2 * pivot - wh
    r2 = pivot + (wh - wl)
    s2 = pivot - (wh - wl)
    r3 = wh + 2 * (pivot - wl)
    s3 = wl - 2 * (wh - pivot)
    
    # Align to 6h timeframe (wait for completed weekly bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 6h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND above weekly pivot AND volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > pivot_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band AND below weekly pivot AND volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < pivot_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian lower band OR below weekly pivot
            if close[i] < lowest_low[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band OR above weekly pivot
            if close[i] > highest_high[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals