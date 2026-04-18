#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d EMA Trend Filter
Hypothesis: Donchian channel breakouts capture institutional moves. Volume confirmation filters false breakouts.
1d EMA filter ensures we only trade in the direction of the daily trend, working in both bull (buy breakouts above upper band in uptrend) and bear (sell breakdowns below lower band in downtrend). Low trade frequency to minimize fee drag.
"""

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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on daily close
    close_1d = df_1d['close'].values
    ema_30 = np.zeros_like(close_1d)
    ema_30[:] = np.nan
    if len(close_1d) >= 30:
        multiplier = 2 / (30 + 1)
        ema_30[29] = np.mean(close_1d[0:30])
        for i in range(30, len(close_1d)):
            ema_30[i] = (close_1d[i] - ema_30[i-1]) * multiplier + ema_30[i-1]
    
    # Align EMA to 4h timeframe
    ema_30_aligned = align_htf_to_ltf(prices, df_1d, ema_30)
    
    # Donchian channel (20-period) on 4h data
    upper = np.zeros_like(high)
    lower = np.zeros_like(low)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(20, len(high)):
        upper[i] = np.max(high[i-20+1:i+1])
        lower[i] = np.min(low[i-20+1:i+1])
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_30_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and daily uptrend
            if close[i] > upper[i] and vol_spike[i] and close[i] > ema_30_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and daily downtrend
            elif close[i] < lower[i] and vol_spike[i] and close[i] < ema_30_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below lower Donchian or volume dies
            if close[i] < lower[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above upper Donchian or volume dies
            if close[i] > upper[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA30"
timeframe = "4h"
leverage = 1.0