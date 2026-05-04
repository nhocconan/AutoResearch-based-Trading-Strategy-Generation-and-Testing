#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation
# Uses Williams Alligator (JAW/TEETH/LIPS) from 6h for trend structure and entry timing
# 1d Elder Ray (Bull/Bear Power) confirms higher timeframe momentum direction
# Volume spike (>1.5x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in both bull and bear by following the higher timeframe trend via Elder Ray
# Alligator provides dynamic support/resistance for entries and exits

name = "6h_WilliamsAlligator_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams Alligator on 6h timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    def smma(arr, window):
        res = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < window:
            return res
        # First value is simple SMA
        res[window-1] = np.mean(arr[:window])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(window, len(arr)):
            res[i] = (res[i-1] * (window-1) + arr[i]) / window
        return res
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift Alligator lines as per Williams specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) +
            # Bull Power > 0 (strong buying pressure) + volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and
                bull_power_aligned[i] > 0 and
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) +
            # Bear Power < 0 (strong selling pressure) + volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and
                  bear_power_aligned[i] < 0 and
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals