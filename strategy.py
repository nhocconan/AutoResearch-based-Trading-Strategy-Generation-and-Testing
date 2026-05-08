# USO 4h_TrendFollowing_ADX14_PlusDI_MinusDI
# Trend-following strategy using ADX and directional indicators to capture trends with proper risk management.
# Works in both bull and bear markets by going long when +DI > -DI and ADX > 25, short when -DI > +DI and ADX > 25.
# Uses 4h timeframe to target 20-50 trades/year, avoiding excessive frequency and fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TrendFollowing_ADX14_PlusDI_MinusDI"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX and directional indicators
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period+1])  # Skip first element (index 0)
        # Subsequent values using Wilder's smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    # Smooth TR, +DM, -DM
    atr = smooth_wilder(tr, 14)
    plus_di_smooth = smooth_wilder(plus_dm, 14)
    minus_di_smooth = smooth_wilder(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, (plus_di_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_di_smooth / atr) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth_wilder(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        
        if position == 0:
            # Enter long: +DI > -DI and ADX > 25 (strong uptrend)
            if plus_di_val > minus_di_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: -DI > +DI and ADX > 25 (strong downtrend)
            elif minus_di_val > plus_di_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX falls below 20 OR -DI crosses above +DI (trend weakening or reversal)
            if adx_val < 20 or minus_di_val > plus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX falls below 20 OR +DI crosses above -DI (trend weakening or reversal)
            if adx_val < 20 or plus_di_val > minus_di_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals