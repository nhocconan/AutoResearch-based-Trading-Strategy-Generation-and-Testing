#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout_Volume_Regime_v2
Camarilla R1/S1 breakout from 1D with volume confirmation and Chop regime filter.
Long: price breaks above R1 (1D) + volume > 1.5x avg + Chop > 61.8 (range)
Short: price breaks below S1 (1D) + volume > 1.5x avg + Chop > 61.8
Exit when price returns to Camarilla midpoint (close) or Chop < 38.2 (trend)
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D Camarilla levels (calculate once) ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla R1, S1, and midpoint (close)
    r1 = pc + 1.1 * (ph - pl) / 12
    s1 = pc - 1.1 * (ph - pl) / 12
    midpoint = pc  # Camarilla close
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
    
    # === Volume ratio (current vs 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Chopiness Index (14) for regime filter ===
    atr_list = []
    for i in range(n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]) if i > 0 else 0, abs(low[i] - close[i-1]) if i > 0 else 0)
        atr_list.append(tr)
    atr = np.array(atr_list)
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    chop = 100 * np.log10(sum_atr_14 / (range_14 + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > R1, volume > 1.5x avg, Chop > 61.8 (range)
            if (close[i] > r1_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < S1, volume > 1.5x avg, Chop > 61.8 (range)
            elif (close[i] < s1_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price < midpoint OR Chop < 38.2 (trend)
            if (close[i] < midpoint_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > midpoint OR Chop < 38.2 (trend)
            if (close[i] > midpoint_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_R1_S1_Breakout_Volume_Regime_v2"
timeframe = "12h"
leverage = 1.0