#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend
Hypothesis: Breakouts at 20-period Donchian channels with 1d EMA trend filter and volume confirmation.
This strategy captures breakouts in the direction of the daily trend, using Donchian channels for
structure and volume to filter false signals. Works in both bull and bear markets by aligning with
higher timeframe trend. Low trade frequency reduces fee drag.
"""

name = "4h_Donchian_Breakout_VolumeTrend"
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
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # === Donchian Channels (20-period) on 4h ===
    # Calculate highest high and lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and daily EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above Donchian high with uptrend and volume
            if (close[i] > highest_high[i] and 
                close[i] > ema34_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price closes below Donchian low with downtrend and volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low (mean reversion)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above Donchian high (mean reversion)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals