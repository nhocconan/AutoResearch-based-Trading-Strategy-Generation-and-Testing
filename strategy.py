#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for ATR-based volatility filter
- Donchian channels calculated from prior 20-period 4h high/low (no look-ahead)
- Breakout logic: long when price closes above upper band with volume spike and ATR > median,
                  short when price closes below lower band with volume spike and ATR > median
- Volatility filter: only trade when current 1d ATR(14) > its 50-period median (avoid low-vol chop)
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: volatility filter avoids ranging markets, Donchian breakouts capture momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) from prior completed periods (shift by 1)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    
    # Rolling max/min on shifted arrays for prior 20 periods
    roll_max = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    roll_min = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    
    # Align 1d ATR and median to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 50)  # Need Donchian, volume MA, and ATR median
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(roll_max[i]) or np.isnan(roll_min[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_median_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian band AND ATR > median AND volume spike
            if close[i] > roll_max[i] and atr_14_aligned[i] > atr_median_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian band AND ATR > median AND volume spike
            elif close[i] < roll_min[i] and atr_14_aligned[i] > atr_median_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Donchian channels or reverse signal
            midpoint = (roll_max[i] + roll_min[i]) / 2.0
            if not np.isnan(midpoint) and close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint of Donchian channels or reverse signal
            midpoint = (roll_max[i] + roll_min[i]) / 2.0
            if not np.isnan(midpoint) and close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0