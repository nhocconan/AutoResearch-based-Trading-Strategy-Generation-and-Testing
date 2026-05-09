#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h close for trend
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day ATR ratio (current vs 20-period average)
    atr_ratio = atr14_1d / pd.Series(atr14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Camarilla levels (R1, S1)
    close_4h_prev = np.roll(close_4h, 1)
    close_4h_prev[0] = close_4h[0]  # avoid NaN on first
    high_4h_prev = np.roll(high_4h, 1)
    high_4h_prev[0] = high_4h[0]
    low_4h_prev = np.roll(low_4h, 1)
    low_4h_prev[0] = low_4h[0]
    
    pivot_4h = (high_4h_prev + low_4h_prev + close_4h_prev) / 3
    range_4h = high_4h_prev - low_4h_prev
    
    r1_4h = pivot_4h + (range_4h * 1.1 / 4)
    s1_4h = pivot_4h - (range_4h * 1.1 / 4)
    
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above 4h EMA20 + ATR ratio > 0.8 (avoid low volatility)
            if (close[i] > r1_aligned[i] and 
                close[i] > ema20_4h_aligned[i] and 
                atr_ratio_aligned[i] > 0.8):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S1 + below 4h EMA20 + ATR ratio > 0.8
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema20_4h_aligned[i] and 
                  atr_ratio_aligned[i] > 0.8):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below 4h EMA20 (trend change)
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 4h EMA20 (trend change)
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals