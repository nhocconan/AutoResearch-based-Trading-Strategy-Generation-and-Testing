#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot bias and volume confirmation
# Uses 20-period Donchian channel breakouts for trend following.
# Weekly pivot direction (from weekly pivot points) provides trend bias: 
# only take long when weekly bias is bullish, short when bearish.
# Volume confirmation requires current volume > 1.5x median of past 20 periods.
# Works in bull markets (breakouts up with bullish bias) and bear markets (breakouts down with bearish bias).
# Target: 50-150 total trades over 4 years = 12-37/year.
# Timeframe: 6h, HTF: 1w for pivot calculation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader method)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Determine weekly bias: bullish if close > pivot, bearish if close < pivot
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, 
                          np.where(weekly_close < weekly_pivot, -1, 0))
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate Donchian channel (20-period) on 6h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(donchian_period, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bias_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + weekly bias bullish + volume confirmation
        if (close[i] > highest_high[i] and
            weekly_bias_aligned[i] == 1 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + weekly bias bearish + volume confirmation
        elif (close[i] < lowest_low[i] and
              weekly_bias_aligned[i] == -1 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Donchian breakout or weekly bias flip
        elif position == 1 and (close[i] < lowest_low[i] or weekly_bias_aligned[i] == -1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or weekly_bias_aligned[i] == 1):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0