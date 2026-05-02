#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses 6h primary timeframe for signal generation with Donchian channel breakouts
# Weekly pivot direction (price above/below weekly pivot) provides higher timeframe bias
# Volume confirmation (2.0x 24-period average) filters for strong participation to reduce false breakouts
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by only trading in direction of weekly trend
# Donchian levels provide objective breakout levels, reducing subjectivity in entries

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = max(lookback, 24) + 1
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper + price > weekly pivot + volume spike
            if close[i] > highest_high[i] and close[i] > weekly_pivot_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + price < weekly pivot + volume spike
            elif close[i] < lowest_low[i] and close[i] < weekly_pivot_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian lower or price < weekly pivot
            if close[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian upper or price > weekly pivot
            if close[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals