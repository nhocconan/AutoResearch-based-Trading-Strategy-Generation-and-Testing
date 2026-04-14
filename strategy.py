#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > weekly pivot S2 (bullish bias) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < weekly pivot R2 (bearish bias) AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel (opposite band)
# Weekly pivot provides directional bias from higher timeframe, reducing counter-trend trades in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 6h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Donchian high AND price > weekly S2 (bullish bias) AND volume confirmation
            if (price > high_20[i] and price > s2_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Donchian low AND price < weekly R2 (bearish bias) AND volume confirmation
            elif (price < low_20[i] and price < r2_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0