#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Control_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Camarilla Pivot Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels (resistance/support)
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 12-period average)
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / np.where(vol_ma12 > 0, vol_ma12, np.nan)
    
    # Chopiness Index for regime filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(sum_tr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop[0:13] = np.nan  # Ensure proper warmup
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ratio_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade in ranging markets (Chop > 61.8)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation in ranging market
            if (close_val > r1_val and    # Break above R1
                vol_ratio_val > 1.5 and   # Volume confirmation
                chop_filter):             # Ranging market
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation in ranging market
            elif (close_val < s1_val and  # Break below S1
                  vol_ratio_val > 1.5 and # Volume confirmation
                  chop_filter):           # Ranging market
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below R1 or chop drops (trending market)
            if (close_val < r1_val) or (not chop_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above S1 or chop drops (trending market)
            if (close_val > s1_val) or (not chop_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals