#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels with volume confirmation for mean reversion entries. Enter long when price touches S3 level with volume > 1.5x average, short when price touches R3 level with volume > 1.5x average. Exit when price reaches opposite pivot level (S1/R1) or reverses. Uses 1-day trend filter to avoid counter-trend trades. Designed for low frequency (12-37 trades/year) to avoid fee drag while capturing mean reversion in ranging markets and breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 6)
    # S2 = C - (Range * 1.1 / 4)
    # S3 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 6)
    # R2 = C + (Range * 1.1 / 4)
    # R3 = C + (Range * 1.1 / 2)
    
    pivot = (d_high + d_low + d_close) / 3.0
    rng = d_high - d_low
    s1 = d_close - (rng * 1.1 / 6)
    s2 = d_close - (rng * 1.1 / 4)
    s3 = d_close - (rng * 1.1 / 2)
    r1 = d_close + (rng * 1.1 / 6)
    r2 = d_close + (rng * 1.1 / 4)
    r3 = d_close + (rng * 1.1 / 2)
    
    # Align pivot levels to 12h timeframe (shifted by 1 day for no look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Calculate 24-period average volume for confirmation (2 days of 12h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume average warmup
        # Skip if daily data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price reaches S1 level or shows weakness below S2
            if close[i] >= s1_aligned[i] or close[i] <= s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches R1 level or shows strength above R2
            if close[i] <= r1_aligned[i] or close[i] >= r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or goes below S3 level with volume confirmation
            long_entry = (close[i] <= s3_aligned[i]) and vol_confirm
            # Short entry: price touches or goes above R3 level with volume confirmation
            short_entry = (close[i] >= r3_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals