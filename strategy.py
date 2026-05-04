#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Weekly pivot (Camarilla) provides structural bias: long above weekly pivot, short below
# Donchian breakout captures momentum in direction of weekly bias
# Volume confirmation ensures breakout validity
# Works in bull markets (breakouts above weekly pivot with volume) and bear markets (breakdowns below weekly pivot with volume)
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
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
    
    # Get 1w data for weekly pivot points (Camarilla)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one week for pivot calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels from prior completed weekly bar
    # Camarilla formulas: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + Range * 1.1/2, S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1, S4 = Pivot - Range * 1.1
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1 / 2.0
    s3_1w = pivot_1w - range_1w * 1.1 / 2.0
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Shift by 1 to use only completed weekly bar (look-ahead protection)
    pivot_1w_shifted = np.roll(pivot_1w, 1)
    r3_1w_shifted = np.roll(r3_1w, 1)
    s3_1w_shifted = np.roll(s3_1w, 1)
    r4_1w_shifted = np.roll(r4_1w, 1)
    s4_1w_shifted = np.roll(s4_1w, 1)
    pivot_1w_shifted[0] = np.nan
    r3_1w_shifted[0] = np.nan
    s3_1w_shifted[0] = np.nan
    r4_1w_shifted[0] = np.nan
    s4_1w_shifted[0] = np.nan
    
    # Align weekly levels to 6h timeframe (available after weekly bar closes)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_shifted)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w_shifted)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w_shifted)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w_shifted)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w_shifted)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND above weekly pivot AND volume spike
            if close[i] > high_max_20[i] and close[i] > pivot_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND below weekly pivot AND volume spike
            elif close[i] < low_min_20[i] and close[i] < pivot_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR below weekly pivot
            if close[i] < low_min_20[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR above weekly pivot
            if close[i] > high_max_20[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals