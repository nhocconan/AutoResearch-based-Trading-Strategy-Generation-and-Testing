#!/usr/bin/env python3
# 6h_1w_1d_alligator_elder_ray_v2
# Hypothesis: 6-hour Elder Ray index with 1-week Alligator filter for trend confirmation
# Works in bull/bear by using Elder Ray for momentum and Alligator for trend filtering
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag

name = "6h_1w_1d_alligator_elder_ray_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Alligator (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Alligator lines: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Get daily data for Elder Ray (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: Alligator alignment (JAW > TEETH > LIPS for uptrend, reverse for downtrend)
        uptrend = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        downtrend = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Long entry: uptrend + Bull Power > 0
        if uptrend and bull_power_aligned[i] > 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: downtrend + Bear Power < 0
        elif downtrend and bear_power_aligned[i] < 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend change or power signal reversal
        elif position == 1 and (not uptrend or bull_power_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or bear_power_aligned[i] >= 0):
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