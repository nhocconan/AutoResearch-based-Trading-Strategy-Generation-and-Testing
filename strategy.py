#!/usr/bin/env python3
"""
4h_ADX_Direction_Trend_v1
Concept: Use ADX(14) to measure trend strength and direction via +DI/-DI crossover.
- Long: ADX > 25 AND +DI crosses above -DI (trending up)
- Short: ADX > 25 AND -DI crosses above +DI (trending down)
- Exit: ADX falls below 20 (trend weakening) OR opposite DI crossover
- Position sizing: 0.25
- Works in bull/bear: ADX filters range markets, DI crossover captures momentum
"""

import numpy as np
import pandas as pd

name = "4h_ADX_Direction_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    tr_smoothed = wilder_smooth(tr, period)
    plus_dm_smoothed = wilder_smooth(plus_dm, period)
    minus_dm_smoothed = wilder_smooth(minus_dm, period)
    
    # Calculate DI+
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period * 2  # Need enough data for smoothing
    
    for i in range(start_idx, n):
        # Skip if any value is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(tr_smoothed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong uptrend (ADX > 25) and +DI crosses above -DI
            if adx[i] > 25 and plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Strong downtrend (ADX > 25) and -DI crosses above +DI
            elif adx[i] > 25 and minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend weakening (ADX < 20) or -DI crosses above +DI
            if adx[i] < 20 or (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend weakening (ADX < 20) or +DI crosses above -DI
            if adx[i] < 20 or (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals