#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Volume_Reversal
Hypothesis: Trade reversals at Camarilla pivot levels on 1d timeframe with volume confirmation on 12h.
In ranging markets, price often reverses at key support/resistance levels (H3/L3).
Go long when price touches L3 with rising volume, short when touches H3 with falling volume.
Works in both bull and bear markets by capturing mean reversion at statistically significant levels.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day: based on previous day's range
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    # Using previous day's values to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first day uses same day
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    range_1d = prev_high_1d - prev_low_1d
    H3 = prev_close_1d + 1.1 * range_1d / 2
    L3 = prev_close_1d - 1.1 * range_1d / 2
    
    # Get 12h data for price and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume_12h > (vol_ma_20_12h * 1.5)
    
    # Align all signals to main timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_12h, volume_expansion)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(volume_expansion_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long when price touches or goes below L3 with volume expansion
        if low[i] <= L3_aligned[i] and volume_expansion_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short when price touches or goes above H3 with volume expansion
        elif high[i] >= H3_aligned[i] and volume_expansion_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit when price moves back toward midline (between H3 and L3)
        elif position == 1 and close[i] > (H3_aligned[i] + L3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < (H3_aligned[i] + L3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        # Hold position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_Reversal"
timeframe = "12h"
leverage = 1.0