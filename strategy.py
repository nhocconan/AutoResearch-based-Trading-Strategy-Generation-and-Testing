#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Donchian(20) breakouts with daily EMA50 trend filter and volume confirmation capture trend continuation moves.
Works in both bull and bear markets by trading only in direction of daily trend, avoiding counter-trend whipsaws.
Targets 20-50 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend turns down
            if close[i] <= low_20[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend turns up
            if close[i] >= high_20[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above upper Donchian + volume confirmation + uptrend
            if (close[i] > high_20[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.30
            # Short: price breaks below lower Donchian + volume confirmation + downtrend
            elif (close[i] < low_20[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.30
    
    return signals