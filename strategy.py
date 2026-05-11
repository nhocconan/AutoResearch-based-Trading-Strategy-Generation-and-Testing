#!/usr/bin/env python3
name = "12h_Donchian20_VolumeBreakout_1dTrend"
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
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian(20) on 12H
    high_max = np.zeros(n)
    low_min = np.zeros(n)
    high_max[:] = np.nan
    low_min[:] = np.nan
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    # 1D EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12H Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > high_max[i] and close[i] > ema50_1d_aligned[i]
        bearish_breakout = close[i] < low_min[i] and close[i] < ema50_1d_aligned[i]
        volume_spike = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            if bullish_breakout and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on bearish breakout or volume dry-up
            if bearish_breakout or volume[i] < 0.7 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bullish breakout or volume dry-up
            if bullish_breakout or volume[i] < 0.7 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals