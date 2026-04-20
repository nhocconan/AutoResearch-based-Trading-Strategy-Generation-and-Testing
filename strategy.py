#!/usr/bin/env python3
# 12h_Pivot_Breakout_VolumeATRFilter_TripleFilter
# Hypothesis: Combines daily pivot point (S1/R1) breakouts with volume confirmation and ATR volatility filter.
# Uses triple confirmation: price must break S1/R1, volume must exceed 2x 20-period average, and ATR must be above its 50-period mean.
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag. Works in both bull/bear markets by capturing breakouts
# from key levels with institutional volume validation and volatility filtering.

name = "12h_Pivot_Breakout_VolumeATRFilter_TripleFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot and S1/R1 from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Standard pivot point calculation
    p = (ph + pl + pc) / 3
    # S1 and R1 using standard formulas
    s1 = 2 * p - ph
    r1 = 2 * p - pl
    
    # Align S1 and R1 to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume filter: volume > 2x 20-period average (stricter)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 2.0)
    
    # ATR filter: only trade when volatility is above average
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > atr_ma50  # Only trade when ATR > its 50-period MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + sufficient volatility
            if close[i] > r1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + sufficient volatility
            elif close[i] < s1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals