#!/usr/bin/env python3
"""
6h_SwingRejection_v1
Hypothesis: Combines price rejection at key swing points with multi-timeframe trend alignment.
Uses 1-day swing high/low detection to identify institutional supply/demand zones.
Enters on price rejection from these zones in the direction of 12h trend.
Targets 15-30 trades/year to minimize fee drag while capturing high-probability setups.
Works in both bull and bear markets by trading rejection of institutional levels.
"""

name = "6h_SwingRejection_v1"
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
    volume = prices['volume'].values
    
    # 1. Daily swing points (institutional supply/demand zones)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Swing high: higher high than previous and next bar
    high_1d = df_1d['high'].values
    swing_high = np.zeros_like(high_1d, dtype=bool)
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = True
    
    # Swing low: lower low than previous and next bar
    low_1d = df_1d['low'].values
    swing_low = np.zeros_like(low_1d, dtype=bool)
    for i in range(1, len(low_1d)-1):
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = True
    
    # Align swing points to 6h timeframe
    swing_high_float = swing_high.astype(float)
    swing_low_float = swing_low.astype(float)
    swing_high_6h = align_htf_to_ltf(prices, df_1d, swing_high_float)
    swing_low_6h = align_htf_to_ltf(prices, df_1d, swing_low_float)
    
    # 2. 12-hour trend filter (institutional timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_6h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 3. Volume confirmation (institutional participation)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4-day average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any critical value is NaN
        if (np.isnan(swing_high_6h[i]) or np.isnan(swing_low_6h[i]) or 
            np.isnan(ema_12h_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price action
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Nearest swing levels (within 0.5% tolerance)
        near_swing_high = swing_high_6h[i] > 0.5
        near_swing_low = swing_low_6h[i] > 0.5
        
        if position == 0:
            # Long setup: rejection from swing low + above 12h EMA + volume
            if (near_swing_low and 
                curr_low <= low[i-1] * 1.005 and  # slight penetration then close back
                curr_close > curr_low and  # bullish close
                curr_close > ema_12h_6h[i] and
                curr_volume > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            
            # Short setup: rejection from swing high + below 12h EMA + volume
            elif (near_swing_high and 
                  curr_high >= high[i-1] * 0.995 and  # slight penetration then close back
                  curr_close < curr_high and  # bearish close
                  curr_close < ema_12h_6h[i] and
                  curr_volume > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: break below swing low or below 12h EMA
            if (near_swing_low and curr_close < low[i-1] * 0.995) or curr_close < ema_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: break above swing high or above 12h EMA
            if (near_swing_high and curr_close > high[i-1] * 1.005) or curr_close > ema_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals