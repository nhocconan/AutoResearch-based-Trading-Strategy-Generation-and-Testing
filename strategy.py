#!/usr/bin/env python3
"""
12h_1d_camarilla_pivot_volume_v1
Hypothesis: Use daily Camarilla pivot levels (R3, S3) with 12h price action and volume confirmation.
Long when price crosses above S3 with volume, short when crosses below R3 with volume.
Camarilla levels provide strong support/resistance; volume confirms breakout authenticity.
Works in both bull/bear markets as it fades extreme moves toward mean (S3/R3).
Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (Camarilla uses previous day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day: use same day's values (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Camarilla levels: S3, S2, S1, R1, R2, R3
    s3 = prev_close - (range_val * 1.1 / 2)
    s2 = prev_close - (range_val * 1.1 / 4)
    s1 = prev_close - (range_val * 1.1 / 6)
    r1 = prev_close + (range_val * 1.1 / 6)
    r2 = prev_close + (range_val * 1.1 / 4)
    r3 = prev_close + (range_val * 1.1 / 2)
    
    # Get 12h data for price reference
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Align Camarilla levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (12 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion complete) or reverses at R3
            if close[i] < s3_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion complete) or reverses at S3
            if close[i] > r3_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price crosses above S3 with volume (bounce from strong support)
            if close[i] > s3_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below R3 with volume (rejection at strong resistance)
            elif close[i] < r3_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals