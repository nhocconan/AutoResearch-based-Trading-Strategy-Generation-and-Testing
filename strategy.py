#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_Volume_Confirm
Hypothesis: Uses Donchian channel (20-period) breakout with volume confirmation and 1d EMA trend filter on 12h timeframe.
Enters long when price breaks above upper Donchian band with volume spike and 1d EMA50 rising.
Enters short when price breaks below lower Donchian band with volume spike and 1d EMA50 falling.
Designed for low-moderate trade frequency (~15-25/year) with trend-following capability in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 slope
    ema50_1d_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema50_1d_aligned[i]) and not np.isnan(ema50_1d_aligned[i-1]):
            ema50_1d_slope[i] = ema50_1d_aligned[i] - ema50_1d_aligned[i-1]
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1d_slope[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and rising 1d EMA50
            if close[i] > upper[i] and vol_spike[i] and ema50_1d_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and falling 1d EMA50
            elif close[i] < lower[i] and vol_spike[i] and ema50_1d_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: break below lower Donchian or 1d EMA50 turns down
            if close[i] < lower[i] or ema50_1d_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: break above upper Donchian or 1d EMA50 turns up
            if close[i] > upper[i] or ema50_1d_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_Volume_Confirm"
timeframe = "12h"
leverage = 1.0