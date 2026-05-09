#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with Elder Ray power and volume confirmation
# Long when green line > red line (bullish alignment) + bull power > 0 + volume > 1.5x average
# Short when red line > green line (bearish alignment) + bear power > 0 + volume > 1.5x average
# Exit when alignment breaks or power becomes negative
# Uses Williams Alligator for trend, Elder Ray for power, volume for confirmation
# Designed to capture strong trends with controlled frequency in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Williams_Alligator_ElderRay_PowerTrend"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Calculate Williams Alligator lines (13,8,5 SMAs with future shifts)
    # Jaw (blue): 13-period SMMA shifted 8 bars
    # Teeth (red): 8-period SMMA shifted 5 bars
    # Lips (green): 5-period SMMA shifted 3 bars
    def smoothed_ma(arr, period):
        # Smoothed Moving Average (SMMA) - similar to RMA/Wilder's smoothing
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma_5 = smoothed_ma(close, 5)
    smma_8 = smoothed_ma(close, 8)
    smma_13 = smoothed_ma(close, 13)
    
    # Apply shifts (Williams Alligator specific)
    lips = np.roll(smma_5, 3)    # 5-period shifted 3 bars forward
    teeth = np.roll(smma_8, 5)   # 8-period shifted 5 bars forward
    jaw = np.roll(smma_13, 8)    # 13-period shifted 8 bars forward
    
    # Calculate Elder Ray power
    # Bull Power = High - EMA(13)
    # Bear Power = EMA(13) - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for SMMA and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: green > red (bullish alignment) + bull power > 0 + volume spike
            if (lips[i] > teeth[i] and 
                bull_power[i] > 0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: red > green (bearish alignment) + bear power > 0 + volume spike
            elif (teeth[i] > lips[i] and 
                  bear_power[i] > 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks or bull power becomes negative
            if (lips[i] <= teeth[i]) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks or bear power becomes negative
            if (teeth[i] <= lips[i]) or (bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals