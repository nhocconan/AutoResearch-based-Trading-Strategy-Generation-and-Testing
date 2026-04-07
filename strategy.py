#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong support/resistance levels.
Price tends to reverse at these levels during ranging markets and break out during trending markets.
Volume surge confirms the breakout/reversal. Works in both bull/bear by taking reversals at S3/R3
in ranging markets and breakouts at S4/R4 in trending markets. Target: 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    range_ = prev_high - prev_low
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)
    s4 = close - (range_ * 1.1 / 2)
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if pivot levels not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or R4
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or S4
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long at S3/S4 with volume surge (bounce)
            if vol_surge and close[i] <= s3_aligned[i] and close[i] > s4_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short at R3/R4 with volume surge (rejection)
            elif vol_surge and close[i] >= r3_aligned[i] and close[i] < r4_aligned[i]:
                position = -1
                signals[i] = -0.25
            # Breakout entries: price breaks S4/R4 with volume surge
            elif vol_surge and close[i] < s4_aligned[i]:
                position = -1
                signals[i] = -0.25
            elif vol_surge and close[i] > r4_aligned[i]:
                position = 1
                signals[i] = 0.25
    
    return signals