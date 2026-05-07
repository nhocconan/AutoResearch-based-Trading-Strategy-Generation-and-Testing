#!/usr/bin/env python3
name = "4h_Donchian_20_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channel (20-period)
    window = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(window - 1, n):
        highest[i] = np.max(high[i-window+1:i+1])
        lowest[i] = np.min(low[i-window+1:i+1])
    
    # 4h volume filter: > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20 - 1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > Donchian upper + trend up + volume
            if (close[i] > highest[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < Donchian lower + trend down + volume
            elif (close[i] < lowest[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < Donchian lower or trend down
            if close[i] < lowest[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > Donchian upper or trend up
            if close[i] > highest[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Donchian(20) captures breakouts from 20-period price channels.
# 1d EMA34 provides higher timeframe trend alignment.
# Volume filter ensures trades occur with market participation.
# Works in both bull and bear markets by following the trend on higher timeframe.
# Position size 0.25 limits risk, targeting 20-40 trades/year to minimize fee drag.