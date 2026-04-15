#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with Elder Ray Power and Volume Confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to define trend direction and filter trades.
# Elder Ray Power (bull/bear power) measures trend strength behind price action.
# Volume confirmation ensures breakouts have participation.
# Works in bull markets (buy when teeth above jaw, power positive) and bear markets (sell when teeth below jaw, power negative).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Future shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Future shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Future shift
    
    # Elder Ray Power
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power.values)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power.values)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            continue
        
        # Long entry: Teeth above Jaw (uptrend) + Bull Power positive + volume confirmation
        if (teeth_aligned[i] > jaw_aligned[i] and
            bull_power_aligned[i] > 0 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Teeth below Jaw (downtrend) + Bear Power negative + volume confirmation
        elif (teeth_aligned[i] < jaw_aligned[i] and
              bear_power_aligned[i] < 0 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Alligator alignment or power divergence
        elif position == 1 and (teeth_aligned[i] < jaw_aligned[i] or bull_power_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (teeth_aligned[i] > jaw_aligned[i] or bear_power_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Williams_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0