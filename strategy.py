#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_Volume_Trend
Hypothesis: On 12h timeframe, Donchian(20) breakout confirmed by 1d trend (EMA34) and volume spike
provides directional momentum signals. Works in both bull and bear markets by following the 1d trend.
Target: 12-30 trades/year per symbol.
"""

name = "12h_1d_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price breaks above Donchian high + 1d uptrend + volume spike
            if not np.isnan(highest_high[i]) and close[i] > highest_high[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + 1d downtrend + volume spike
            elif not np.isnan(lowest_low[i]) and close[i] < lowest_low[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or trend changes
            if not np.isnan(lowest_low[i]) and close[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or trend changes
            if not np.isnan(highest_high[i]) and close[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals