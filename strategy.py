#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume
- Pivot point bounce (R1/S1) with volume confirmation for mean reversion
- Breakout at R2/S2 with volume surge for trend continuation
- Works in both bull/bear: mean reversion in range, breakout in trend
- Uses 1d Pivot for 12h timeframe
- Target: 50-150 trades over 4 years (12-37/year)
- Size: 0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Pivot levels (R1, S1, R2, S2)
    r1 = pivot + (range_val * 1.0 / 2)  # First resistance
    s1 = pivot - (range_val * 1.0 / 2)  # First support
    r2 = pivot + range_val              # Second resistance
    s2 = pivot - range_val              # Second support
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
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
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(r2_val) or 
            np.isnan(s2_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion at R1/S1
            if close_val < r1_val and close_val > r1_val * 0.998 and vol_ratio_val > 1.5:
                # Near R1, short with volume
                signals[i] = -0.25
                position = -1
            elif close_val > s1_val and close_val < s1_val * 1.002 and vol_ratio_val > 1.5:
                # Near S1, long with volume
                signals[i] = 0.25
                position = 1
            # Breakout at R2/S2
            elif close_val > r2_val and vol_ratio_val > 2.0:
                # Strong break above R2
                signals[i] = 0.25
                position = 1
            elif close_val < s2_val and vol_ratio_val > 2.0:
                # Strong break below S2
                signals[i] = -0.25
                position = -1
        
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