#!/usr/bin/env python3
name = "6h_WilliamsAlligator_ElderRay_1dTrend"
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
    
    # === 1d Williams Alligator ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Smoothed medians: SMA of median price (H+L)/2
    median_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d Elder Ray (Bull/Bear Power) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Alligator alignment + Elder Ray confirmation
            # Bullish: Lips > Teeth > Jaw AND Bull Power > 0
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Bearish: Lips < Teeth < Jaw AND Bear Power < 0
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  bear_power_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator reverses OR Bear Power turns negative
            if (lips_aligned[i] < teeth_aligned[i] or
                bear_power_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses OR Bull Power turns positive
            if (lips_aligned[i] > teeth_aligned[i] or
                bull_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals