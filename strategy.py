#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Donchian(20) breakout captures momentum, weekly pivot provides HTF bias (from 1w timeframe),
# volume confirmation ensures institutional participation. Designed for low trade frequency
# (12-37/year) to minimize fee drag. Works in both bull and bear markets by using
# weekly pivot as regime filter (above pivot = bullish bias, below = bearish bias).

name = "6h_Donchian20_WeeklyPivot_Volume_v1"
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
    
    # Get 1w data for weekly pivot (HTF bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # We'll use the weekly pivot as the main bias indicator
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot for each weekly bar
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Align weekly pivot to 6h timeframe (wait for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 60  # max(20 for Donchian + 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian(20) high AND above weekly pivot (bullish bias) + volume spike
            if (close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) low AND below weekly pivot (bearish bias) + volume spike
            elif (close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian(20) low OR below weekly pivot (bearish bias shift)
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian(20) high OR above weekly pivot (bullish bias shift)
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals