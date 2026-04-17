#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1
Camarilla R1/S1 breakout from daily pivot + volume spike + chop regime filter.
Long: price breaks above R1 with volume spike in non-choppy market.
Short: price breaks below S1 with volume spike in non-choppy market.
Exit when price returns to central pivot (PP).
Designed to capture institutional breakouts with volume confirmation.
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
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    pp = (ph + pl + pc) / 3.0
    r1 = pc + ((ph - pl) * 1.1 / 12)
    s1 = pc - ((ph - pl) * 1.1 / 12)
    
    # Align to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Spike Detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Choppiness Index Regime Filter (14-period) ===
    atr1 = pd.Series(np.maximum(high - low, 
                                np.maximum(np.abs(high - np.roll(close, 1)), 
                                         np.abs(low - np.roll(close, 1))))).rolling(
        window=14, min_periods=14).mean().values
    atr1[0] = high[0] - low[0]
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    chop = 100 * np.log10(atr1 * 14 / (range_14 + 1e-10)) / np.log10(14)
    chop[range_14 == 0] = 50  # neutral when no range
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, not choppy (trending)
            if (close[i] > r1_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                chop[i] < 61.8):  # trending regime
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, not choppy (trending)
            elif (close[i] < s1_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  chop[i] < 61.8):  # trending regime
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot point
        elif position == 1:
            # Exit long when price returns to or below pivot
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to or above pivot
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0