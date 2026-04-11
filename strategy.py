# 4h_1d_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot level touch with 1d volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (S3/S4 for long, R3/R4 for short) act as strong
# support/resistance. Price touching these levels with above-average 1d volume indicates
# institutional interest and potential reversal. Works in both bull (buy dips) and bear
# (sell rallies) by fading extreme moves at key levels. Low trade frequency expected.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1,S2,S3,S4 and R1,R2,R3,R4
    s1 = close_1d - (range_1d * 1.0 / 6)
    s2 = close_1d - (range_1d * 2.0 / 6)
    s3 = close_1d - (range_1d * 3.0 / 6)
    s4 = close_1d - (range_1d * 4.0 / 6)
    r1 = close_1d + (range_1d * 1.0 / 6)
    r2 = close_1d + (range_1d * 2.0 / 6)
    r3 = close_1d + (range_1d * 3.0 / 6)
    r4 = close_1d + (range_1d * 4.0 / 6)
    
    # Use S3/S4 for long, R3/R4 for short (more extreme levels)
    long_level = s3  # More conservative than S4
    short_level = r3  # More conservative than R4
    
    # Align Camarilla levels to 4h
    long_level_aligned = align_htf_to_ltf(prices, df_1d, long_level)
    short_level_aligned = align_htf_to_ltf(prices, df_1d, short_level)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if np.isnan(long_level_aligned[i]) or np.isnan(short_level_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Price touching Camarilla levels (with small buffer)
        # Long: touches or goes below S3 level
        touch_long = low[i] <= long_level_aligned[i] * 1.001  # 0.1% buffer
        # Short: touches or goes above R3 level
        touch_short = high[i] >= short_level_aligned[i] * 0.999  # 0.1% buffer
        
        # Entry conditions
        # Long: Price touches S3 AND volume confirmation AND not already long
        if touch_long and vol_confirm and position != 1:
            # Additional check: ensure we didn't already touch in previous bar (avoid chattering)
            if i == 30 or low[i-1] > long_level_aligned[i-1] * 1.001:
                position = 1
                signals[i] = 0.25
        # Short: Price touches R3 AND volume confirmation AND not already short
        elif touch_short and vol_confirm and position != -1:
            # Additional check: ensure we didn't already touch in previous bar
            if i == 30 or high[i-1] < short_level_aligned[i-1] * 0.999:
                position = -1
                signals[i] = -0.25
        # Exit: Price moves back toward midpoint (mean reversion)
        elif position == 1 and close[i] >= (long_level_aligned[i] + short_level_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= (long_level_aligned[i] + short_level_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals