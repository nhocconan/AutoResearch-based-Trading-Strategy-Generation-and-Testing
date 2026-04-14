#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(15) high AND weekly pivot is bullish AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(15) low AND weekly pivot is bearish AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel (opposite band)
# Weekly pivot uses prior week's high/low/close to calculate pivot and S1/R1 levels
# This strategy targets trending moves aligned with weekly structure, reducing counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 6h (15-period high/low)
    high_15 = pd.Series(high).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Calculate weekly pivot points
    # Pivot = (H + L + C)/3, S1 = 2*P - H, R1 = 2*P - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    s1_1w = 2 * pivot_1w - high_1w  # Support 1
    r1_1w = 2 * pivot_1w - low_1w   # Resistance 1
    
    # Determine weekly bias: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    
    # Align weekly data to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_15[i]) or np.isnan(low_15[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Donchian high + weekly bullish + volume confirmation
            if (price > high_15[i] and weekly_bullish_aligned[i] > 0.5 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Donchian low + weekly bearish + volume confirmation
            elif (price < low_15[i] and weekly_bearish_aligned[i] > 0.5 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_15[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_15[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0