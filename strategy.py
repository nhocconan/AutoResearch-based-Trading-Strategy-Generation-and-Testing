#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Price breakout above/below 12-hour Donchian channels, filtered by 1-day EMA trend direction and volume confirmation, captures sustained trends in both bull and bear markets while avoiding false breakouts. Uses tight entry conditions to limit trades and reduce fee drag.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
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
    
    # 12h Donchian Channel (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donch_len-1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Get 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_len = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * (ema_len - 1)) / (ema_len + 1)
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + 1d EMA uptrend + volume spike
            if close[i] > upper[i] and close[i] > ema_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 1d EMA downtrend + volume spike
            elif close[i] < lower[i] and close[i] < ema_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower Donchian or trend reverses
            if close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper Donchian or trend reverses
            if close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals