#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# Combines Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# Elder Ray Power (bull/bear power) for momentum strength,
# and volume surge confirmation to filter false breakouts.
# Works in bull markets (teeth above jaw, power > 0) and bear markets (teeth below jaw, power < 0).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator (SMMA-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMMA(13,8), SMMA(8,5), SMMA(13,3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Align Alligator lines to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align Elder Ray to 4h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike: current > 2.0 * median of last 20 periods
    volume_median = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_median[i] = np.median(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_median[i])):
            continue
        
        # Long entry: Teeth above Jaw (uptrend) + Bull Power > 0 + Volume spike
        if (teeth_aligned[i] > jaw_aligned[i] and
            bull_power_aligned[i] > 0 and
            volume[i] > 2.0 * volume_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Teeth below Jaw (downtrend) + Bear Power < 0 + Volume spike
        elif (teeth_aligned[i] < jaw_aligned[i] and
              bear_power_aligned[i] < 0 and
              volume[i] > 2.0 * volume_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Alligator alignment or power crosses zero
        elif position == 1 and (teeth_aligned[i] < jaw_aligned[i] or bull_power_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (teeth_aligned[i] > jaw_aligned[i] or bear_power_aligned[i] >= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Williams_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0