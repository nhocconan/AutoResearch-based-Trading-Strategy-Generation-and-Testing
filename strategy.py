#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray Power + Volume Confirmation
# Long when: Alligator bullish (jaw < teeth < lips) AND 1d Bull Power > 0 AND volume > 1.5x 20 EMA
# Short when: Alligator bearish (jaw > teeth > lips) AND 1d Bear Power < 0 AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams Alligator identifies trend structure via smoothed medians; Elder Ray measures bull/bear power relative to EMA13; volume confirms momentum.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends, with range avoidance via Alligator convergence.

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
    
    # Get 6h data ONCE before loop for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate 6h Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # All lines are SMMA (smoothed moving average) of median price
    median_6h = (high + low) / 2
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    median_6h = (high_6h + low_6h) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_6h, 13)  # Blue line
    teeth = smma(median_6h, 8)   # Red line
    lips = smma(median_6h, 5)    # Green line
    
    # Alligator relationships: Bullish when Lips > Teeth > Jaw, Bearish when Jaw > Teeth > Lips
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (jaw > teeth) & (teeth > lips)
    
    # Align 6h Alligator to prices timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_6h, bullish_alligator.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_6h, bearish_alligator.astype(float))
    
    # Get 1d data for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to prices timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND 1d Bull Power > 0 AND volume spike
            if (bullish_aligned[i] > 0.5 and 
                bull_power_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND 1d Bear Power < 0 AND volume spike
            elif (bearish_aligned[i] > 0.5 and 
                  bear_power_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0
            if (bearish_aligned[i] > 0.5 or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power >= 0
            if (bullish_aligned[i] > 0.5 or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals