#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
Donchian channels capture volatility-based breakouts. 1d EMA50 filters trend direction.
Volume spike confirms institutional participation. 12h timeframe targets 12-37 trades/year
with discrete sizing 0.25 to minimize fee drag. Works in both bull/bear via trend filter.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower bands (20-period)
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1d, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > upper Donchian AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < lower Donchian AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < lower Donchian OR price < 1d EMA50
                if close[i] < lower_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > upper Donchian OR price > 1d EMA50
                if close[i] > upper_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0