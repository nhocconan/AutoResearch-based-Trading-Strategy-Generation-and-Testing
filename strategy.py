#!/usr/bin/env python3
"""
4h_Cam_Pivot_R1S1_Breakout_Volume_ChopFilter_v1
Concept: Camarilla pivot R1/S1 breakout with volume confirmation and chop filter.
- Long when price breaks above previous day's R1 with volume > 1.8x avg and chop > 61.8
- Short when price breaks below previous day's S1 with volume > 1.8x avg and chop > 61.8
- Exit when price returns to previous day's close (mean reversion)
- Chop filter avoids false breakouts in ranging markets
- Conservative sizing (0.25) to manage drawdown
- Works in bull (breakouts) and bear (mean reversion to daily close)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Cam_Pivot_R1S1_Breakout_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    c_1d = pivot_1d  # Close level for exit
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    r1_prev = np.roll(r1_1d, 1)
    s1_prev = np.roll(s1_1d, 1)
    c_prev = np.roll(c_1d, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    c_prev[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    c_aligned = align_htf_to_ltf(prices, df_1d, c_prev)
    
    # === 4h: Chopiness Index (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_sum14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum14 / (atr14 * 14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((atr14 > 0) & (tr_sum14 > 0), chop, 50.0)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        c_val = c_aligned[i]
        chop_val = chop[i]
        vol_ratio_val = vol_ratio[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(c_val) or 
            np.isnan(chop_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and in choppy market
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.8
            chop_filter = chop_val > 61.8  # Choppy/ranging market
            
            if breakout_long and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and in choppy market
            elif close_val < s1_val and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below previous day's close
            if close_val <= c_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above previous day's close
            if close_val >= c_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals