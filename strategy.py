#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_1dTrend_VolumeFilter
Hypothesis: Trade Donchian(20) breakouts on 4h timeframe with 1d EMA trend filter and volume confirmation.
Uses tight entry conditions (price must break above/below 20-period high/low with trend alignment and volume > 1.5x 20-period EMA volume).
Designed to work in both bull and bear markets by following daily trend direction. Targets 20-50 trades/year to minimize fee drag.
"""

name = "4h_Donchian_Breakout_20_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Donchian Channels (20-period) ===
    # Calculate rolling max/min directly on price arrays
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(20, n):
        highest[i] = np.max(high[i-20:i])
        lowest[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_4h[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-period high with uptrend and volume
            if (close[i] > highest[i] and 
                close[i] > ema34_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-period low with downtrend and volume
            elif (close[i] < lowest[i] and 
                  close[i] < ema34_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 20-period low (reversal)
            if close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above 20-period high (reversal)
            if close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals