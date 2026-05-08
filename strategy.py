#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Williams_Alligator_ElderRay_Combo"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    close_1d = df_1d['close'].values
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full(len(data), np.nan)
        sma = np.mean(data[:period])
        result[period-1] = sma
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Jaw, Teeth, Lips aligned to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Bull Power and Bear Power aligned to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips above Teeth (bullish alignment) AND Bull Power > 0 (strong buying pressure)
            long_cond = (lips_aligned[i] > teeth_aligned[i]) and (bull_power_aligned[i] > 0)
            
            # Short entry: Lips below Teeth (bearish alignment) AND Bear Power < 0 (strong selling pressure)
            short_cond = (lips_aligned[i] < teeth_aligned[i]) and (bear_power_aligned[i] < 0)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips cross below Teeth (trend reversal) OR Bull Power <= 0 (loss of buying pressure)
            if (lips_aligned[i] <= teeth_aligned[i]) or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips cross above Teeth (trend reversal) OR Bear Power >= 0 (loss of selling pressure)
            if (lips_aligned[i] >= teeth_aligned[i]) or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator defines trend (Lips-Teeth-Jaw alignment) while Elder Ray measures market power (Bull/Bear).
# Long when Lips > Teeth (bullish alignment) AND Bull Power > 0 (buyers in control).
# Short when Lips < Teeth (bearish alignment) AND Bear Power < 0 (sellers in control).
# Exits when alignment breaks or power dissipates.
# Works in bull markets (trend following) and bear markets (counter-trend reversals via power shifts).
# Williams Alligator uses SMMA (Smoothed Moving Average) for smoother trend identification.
# Elder Ray uses EMA13 for responsive power measurement.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.