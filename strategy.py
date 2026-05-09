#!/usr/bin/env python3
# 12h_WilliamsAlligator_ElderRay_Trend
# Hypothesis: Williams Alligator identifies trend direction (jaw-teeth-lips alignment),
# Elder Ray (Bull/Bear Power) measures trend strength, combined with 1-week trend filter.
# Works in bull/bear: Alligator filters sideways markets, Elder Ray confirms momentum,
# weekly trend ensures alignment with higher timeframe direction.
# Uses Williams Alligator (SMAs: 13,8,5) and Elder Ray (EMA13) from 1d timeframe.

name = "12h_WilliamsAlligator_ElderRay_Trend"
timeframe = "12h"
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
    
    # Calculate Williams Alligator and Elder Ray from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs
    def sma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period-1, len(arr)):
                result[i] = np.mean(arr[i-period+1:i+1])
        return result
    
    jaw = sma(close_1d, 13)   # Blue line
    teeth = sma(close_1d, 8)  # Red line
    lips = sma(close_1d, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def ema(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            multiplier = 2 / (period + 1)
            result[period-1] = np.mean(arr[0:period])
            for i in range(period, len(arr)):
                result[i] = (arr[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema13 = ema(close_1d, 13)
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier) + (ema_50_1w[i-1] * (1 - multiplier))
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure 1d indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND price > weekly EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power < 0 AND price < weekly EMA50
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  bear_power_aligned[i] < 0 and
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator misaligned OR Bull Power <= 0
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and bull_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator misaligned OR Bear Power >= 0
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and bear_power_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals