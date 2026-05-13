#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) combination.
# Uses 6h Alligator (Jaw/Teeth/Lips) for trend direction and 1d Elder Ray for momentum confirmation.
# Long when price > Alligator Lips and Bull Power > 0; Short when price < Alligator Lips and Bear Power < 0.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by requiring alignment between 6h trend and 1d momentum.

name = "6h_WilliamsAlligator_1dElderRay_v1"
timeframe = "6h"
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
    
    # Calculate 6h Williams Alligator (Jaw, Teeth, Lips) - HTF
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    close_6h = df_6h['close'].values
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Calculate 1d Elder Ray (Bull Power, Bear Power) - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # EMA13 of close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema13_1d   # Bear Power = Low - EMA13
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Warmup for Alligator and Elder Ray
        # Skip if any required data is NaN
        if (np.isnan(lips_6h_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Alligator Lips AND Bull Power > 0
            if (close[i] > lips_6h_aligned[i] and 
                bull_power_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Alligator Lips AND Bear Power < 0
            elif (close[i] < lips_6h_aligned[i] and 
                  bear_power_1d_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Alligator Lips OR Bull Power <= 0
            if (close[i] < lips_6h_aligned[i]) or (bull_power_1d_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Alligator Lips OR Bear Power >= 0
            if (close[i] > lips_6h_aligned[i]) or (bear_power_1d_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals