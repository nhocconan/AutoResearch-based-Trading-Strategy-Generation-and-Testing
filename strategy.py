#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakouts aligned with weekly pivot direction and volume confirmation capture institutional breakouts in both bull and bear markets. Weekly pivot provides macro bias, reducing false breakouts. Volume ensures commitment. Designed for 6h timeframe to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot (from weekly OHLC) for directional bias
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pw = (high_1w + low_1w + close_1w) / 3
    rw = high_1w - low_1w
    r1w = pw + rw * 1.1 / 12  # Weekly R1
    s1w = pw - rw * 1.1 / 12  # Weekly S1
    
    # Align weekly levels to 6h
    pw_aligned = align_htf_to_ltf(prices, df_1w, pw)
    r1w_aligned = align_htf_to_ltf(prices, df_1w, r1w)
    s1w_aligned = align_htf_to_ltf(prices, df_1w, s1w)
    
    # Daily trend filter (EMA34 on 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on 6h
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(20, n):
        highest[i] = np.max(high[i-20:i])
        lowest[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure Donchian and EMA ready
    
    for i in range(start_idx, n):
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(pw_aligned[i]) or np.isnan(r1w_aligned[i]) or 
            np.isnan(s1w_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout above weekly R1 with volume and daily uptrend
            if (high[i] > highest[i] and close[i] > r1w_aligned[i] and 
                vol_spike[i] and close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below weekly S1 with volume and daily downtrend
            elif (low[i] < lowest[i] and close[i] < s1w_aligned[i] and 
                  vol_spike[i] and close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly pivot or trend turns down
            if (close[i] < pw_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly pivot or trend turns up
            if (close[i] > pw_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0