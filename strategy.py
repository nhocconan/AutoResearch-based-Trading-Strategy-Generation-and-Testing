# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1d_Pivot_Zone_Reversal_V1
Hypothesis: Price often reverses near daily pivot zones (R1/S1) during low volatility (chop>60).
In ranging markets (chop>60), we fade extremes; in trending markets (chop<40), we breakout.
Timeframe: 6h (primary), HTF: 1d for pivot and chop regime.
Target: 12-37 trades/year (50-150 over 4 years). Size: 0.25.
Works in bull/bear via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_Zone_Reversal_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Standard pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + (range_val * 1.0 / 2)  # R1 = P + (H-L)/2
    s1 = pivot - (range_val * 1.0 / 2)  # S1 = P - (H-L)/2
    r2 = pivot + range_val              # R2 = P + (H-L)
    s2 = pivot - range_val              # S2 = P - (H-L)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === Chopiness Index (1d) for regime filter ===
    # True range
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR(14)
    atr = np.zeros_like(tr)
    atr[:14] = np.nan
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of absolute returns over 14 periods
    abs_returns = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_ret = np.zeros_like(abs_returns)
    for i in range(len(abs_returns)):
        if i < 14:
            sum_abs_ret[i] = np.sum(abs_returns[:i+1])
        else:
            sum_abs_ret[i] = np.sum(abs_returns[i-13:i+1])
    
    # Chopiness index formula
    chop = np.full_like(close_1d, 50.0)  # neutral default
    for i in range(len(close_1d)):
        if sum_abs_ret[i] > 0 and not np.isnan(atr[i]) and atr[i] > 0:
            chop[i] = 100 * np.log10(sum_abs_ret[i] / (atr[i] * 14)) / np.log10(14)
    
    # Align chop to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.full_like(volume, 1.0), where=vol_ma20>0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(chop_val) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime
            is_ranging = chop_val > 60
            is_trending = chop_val < 40
            
            if is_ranging:
                # In ranging markets: fade at S1/R1 with volume confirmation
                if close_val <= s1_aligned[i] and vol_ratio_val > 1.8:
                    # Long near S1 support
                    signals[i] = 0.25
                    position = 1
                elif close_val >= r1_aligned[i] and vol_ratio_val > 1.8:
                    # Short near R1 resistance
                    signals[i] = -0.25
                    position = -1
            elif is_trending:
                # In trending markets: breakout continuation with volume
                if close_val > r2_aligned[i] and vol_ratio_val > 2.0:
                    # Break above R2 -> long
                    signals[i] = 0.25
                    position = 1
                elif close_val < s2_aligned[i] and vol_ratio_val > 2.0:
                    # Break below S2 -> short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if chop_val > 65:  # Market became ranging
                signals[i] = 0.0
                position = 0
            elif close_val < pivot_aligned[i]:  # Price returned below pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Hold position
        
        elif position == -1:
            # Short exit conditions
            if chop_val > 65:  # Market became ranging
                signals[i] = 0.0
                position = 0
            elif close_val > pivot_aligned[i]:  # Price returned above pivot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Hold position
    
    return signals