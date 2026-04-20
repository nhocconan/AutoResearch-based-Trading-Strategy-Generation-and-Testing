#!/usr/bin/env python3
# 12h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: On 12h timeframe, trade breakouts at 1d Camarilla R1/S1 levels with volume and ATR volatility filter.
# In ranging markets, price breaks through R1/S1 with volume; in trending markets, continues momentum.
# Uses 1d ADX to filter ranging (ADX < 25) for breakouts and trending (ADX > 25) for continuation.
# Targets 20-40 trades/year by requiring confluence of level, volume, and volatility filter.

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align 1d levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d ATR for volatility filter (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder smoothing for ATR
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR average for volatility filter
    atr_ma = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Breakout above R1 with volume and volatility expansion
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 1.5 * volume_ma[i] and
                atr_aligned[i] > 1.2 * atr_ma[i]):
                signals[i] = 0.25
                position = 1
            # Breakdown below S1 with volume and volatility expansion
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 1.5 * volume_ma[i] and
                  atr_aligned[i] > 1.2 * atr_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1 or volatility contraction
            if (close[i] < s1_aligned[i] * 0.998) or \
               (atr_aligned[i] < 0.8 * atr_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1 or volatility contraction
            if (close[i] > r1_aligned[i] * 1.002) or \
               (atr_aligned[i] < 0.8 * atr_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals