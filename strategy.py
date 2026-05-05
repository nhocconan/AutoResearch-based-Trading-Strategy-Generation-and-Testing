#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price > Alligator Jaw AND 1w close > 1w EMA50 AND volume > 2.0x 20 EMA
# Short when price < Alligator Jaw AND 1w close < 1w EMA50 AND volume > 2.0x 20 EMA
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Uses Jaw as main trend filter; price must be outside Jaw's mouth to enter.
# 1w EMA50 filters higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms momentum breakout.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2.0
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev_SMMA*(period-1) + Current_Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines
    jaw = smma(median_12h, 13)  # Jaw: 13-period SMMA
    jaw = np.roll(jaw, 8)       # Smoothed 8 bars forward
    
    teeth = smma(median_12h, 8)   # Teeth: 8-period SMMA
    teeth = np.roll(teeth, 5)     # Smoothed 5 bars forward
    
    lips = smma(median_12h, 5)    # Lips: 5-period SMMA
    lips = np.roll(lips, 3)       # Smoothed 3 bars forward
    
    # Align Alligator Jaw to prices timeframe (using Jaw as main trend indicator)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 12h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Alligator Jaw AND 1w uptrend AND volume spike
            if (close[i] > jaw_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Alligator Jaw AND 1w downtrend AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Alligator Jaw OR 1w trend changes to downtrend
            if (close[i] < jaw_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Alligator Jaw OR 1w trend changes to uptrend
            if (close[i] > jaw_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals