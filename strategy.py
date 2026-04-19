#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_With_Volume_Confirm
Hypothesis: Fade at Camarilla R3/S3 levels with volume exhaustion signals. 
In ranging markets (common in 2025), price often reverses at these levels. 
Volume drying up on approach indicates lack of conviction for breakout.
Works in both bull/bear as it fades extremes rather than following trends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Camarilla_R3S3_Fade_With_Volume_Confirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    rang = ph - pl
    r3 = pc + (rang * 1.1 / 4)
    s3 = pc - (rang * 1.1 / 4)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume exhaustion: current volume < 50% of 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (volume_ma * 0.5)
    
    # Optional: avoid extreme volatility (use ATR-based filter)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid extremely volatile periods (ATR > 2 * 50-period average)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr < (atr_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        vol_exhaust = volume_exhaustion[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price approaches S3 from below with volume exhaustion
            if (low[i] <= s3_aligned[i] * 1.005 and  # Within 0.5% of S3
                close[i] > s3_aligned[i] and         # But closed above it (rejection)
                vol_exhaust and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price approaches R3 from above with volume exhaustion
            elif (high[i] >= r3_aligned[i] * 0.995 and  # Within 0.5% of R3
                  close[i] < r3_aligned[i] and          # But closed below it (rejection)
                  vol_exhaust and vol_ok):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price moves back below S3 or reaches R3 (mean reversion target)
            if close[i] < s3_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price moves back above R3 or reaches S3
            if close[i] > r3_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals