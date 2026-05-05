#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d Volume Spike Filter
# - Uses weekly Camarilla levels (R3/S3, R4/S4) from prior completed week
# - Long when price breaks above R4 with 1d volume > 1.5x 20-period average
# - Short when price breaks below S4 with 1d volume > 1.5x 20-period average
# - Exit when price returns to R3/S3 levels or volume drops below average
# - Weekly pivot provides structural HTF bias, volume confirms momentum
# - Designed for low-frequency, high-conviction breakouts in both bull/bear markets
# - Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25

name = "6h_WeeklyCamarilla_VolumeBreakout"
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
    
    # Get weekly data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for meaningful weekly pivots
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.500
    # R3 = C + Range * 1.250
    # S3 = C - Range * 1.250
    # S4 = C - Range * 1.500
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.500
    r3_1w = close_1w + range_1w * 1.250
    s3_1w = close_1w - range_1w * 1.250
    s4_1w = close_1w - range_1w * 1.500
    
    # Align weekly Camarilla levels to 6h
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get daily data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    vol_avg_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume average to 6h
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current 6h bar values
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        if position == 0:
            # Long: Break above R4 with volume confirmation
            if (curr_high > r4_1w_aligned[i] and 
                curr_vol > vol_avg_20_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 with volume confirmation
            elif (curr_low < s4_1w_aligned[i] and 
                  curr_vol > vol_avg_20_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Return to R3 or volume drops
            if (curr_low <= r3_1w_aligned[i] or 
                curr_vol < vol_avg_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Return to S3 or volume drops
            if (curr_high >= s3_1w_aligned[i] or 
                curr_vol < vol_avg_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals