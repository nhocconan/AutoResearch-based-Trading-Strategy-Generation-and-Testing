#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_Volume
- Breakout at R1/S1 with volume confirmation
- Uses 1-day Camarilla for 6h timeframe
- Works in both bull/bear: breakout continuation in trend
- Target: 50-150 trades over 4 years (12-37/year)
- Size: 0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Previous Day's High/Low/Close for Camarilla ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead (use current day's values)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot point (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: R1, S1, R4, S4
    r1 = pivot + (range_val * 1.1 / 12)  # ~9.16% of range
    s1 = pivot - (range_val * 1.1 / 12)
    r4 = pivot + (range_val * 1.1 / 2)   # Breakout level
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(r4_val) or 
            np.isnan(s4_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout at R1/S1 with volume confirmation
            if close_val > r1_val and vol_ratio_val > 1.8:
                # Break above R1
                signals[i] = 0.25
                position = 1
            elif close_val < s1_val and vol_ratio_val > 1.8:
                # Break below S1
                signals[i] = -0.25
                position = -1
            # Optional: Fade at extremes (R4/S4) in ranging markets
            elif close_val > r4_val and vol_ratio_val < 0.8:
                # Fade at R4 in low volume
                signals[i] = -0.25
                position = -1
            elif close_val < s4_val and vol_ratio_val < 0.8:
                # Fade at S4 in low volume
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: return to pivot or stop at S1
            if close_val < pivot_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot or stop at R1
            if close_val > pivot_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals