#!/usr/bin/env python3
"""
6h_WilliamsAlligator_ElderRay_Trend
Hypothesis: Combines Williams Alligator (trend detection) with Elder Ray (bull/bear power) on 6h timeframe.
Williams Alligator uses SMAs of 13, 8, 5 periods to identify trend direction and strength.
Elder Ray measures bull power (high - EMA13) and bear power (EMA13 - low) to confirm trend strength.
In bull markets: Go long when Alligator is bullish (jaws < teeth < lips) and bull power is rising.
In bear markets: Go short when Alligator is bearish (jaws > teeth > lips) and bear power is rising.
Works in both regimes by following the trend direction confirmed by Elder Ray.
Uses volume confirmation to avoid false breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_WilliamsAlligator_ElderRay_Trend"
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
    
    # Get 1d data for Williams Alligator and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator SMAs (13, 8, 5)
    def sma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period-1, len(arr)):
                result[i] = np.mean(arr[i-period+1:i+1])
        return result
    
    # Alligator lines: Jaws (13), Teeth (8), Lips (5)
    jaws = sma(close_1d, 13)
    teeth = sma(close_1d, 8)
    lips = sma(close_1d, 5)
    
    # Align Alligator lines to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
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
    bear_power = ema13 - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions
        alligator_bullish = (jaws_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        alligator_bearish = (jaws_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Enter long: Alligator bullish AND rising bull power AND volume spike
            if (alligator_bullish and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish AND rising bear power AND volume spike
            elif (alligator_bearish and 
                  bear_power_aligned[i] > bear_power_aligned[i-1] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR bull power falls
            if not alligator_bullish or bull_power_aligned[i] < bull_power_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR bear power falls
            if not alligator_bearish or bear_power_aligned[i] < bear_power_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals