#!/usr/bin/env python3
# 6h_1d_elder_ray_reversal_with_volume
# Hypothesis: Elder Ray (Bull/Bear Power) on 1d with volume confirmation for reversals at extremes
# Works in bull/bear by capturing exhaustion moves when price extends beyond 13EMA with volume divergence.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.

name = "6h_1d_elder_ray_reversal_with_volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_ser = pd.Series(close_1d)
    ema13 = close_ser.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_1d - ema13  # Bull Power: High - EMA13
    bear_power = low_1d - ema13   # Bear Power: Low - EMA13
    
    # Volume moving average for confirmation
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_expansion = df_1d['volume'].values > (vol_ma_1d * 1.5)
    
    # Align Elder Ray and volume expansion to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: Bear Power shows exhaustion (less negative) with volume expansion
        # and price near support
        if (bear_power_aligned[i] > bear_power_aligned[i-1] * 1.2 and  # Bear Power improving
            vol_expansion_aligned[i] > 0.5 and                       # Volume expansion
            close[i] < close[i-1] * 1.02 and                         # Not chasing extended moves
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: Bull Power shows exhaustion (less positive) with volume expansion
        elif (bull_power_aligned[i] < bull_power_aligned[i-1] * 0.8 and  # Bull Power weakening
              vol_expansion_aligned[i] > 0.5 and                       # Volume expansion
              close[i] > close[i-1] * 0.98 and                         # Not chasing extended moves
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or power crosses zero
        elif position == 1 and bull_power_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear_power_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals