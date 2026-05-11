#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter"
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter (50-period SMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    weekly_uptrend = close > sma_50_1w_aligned
    
    # Volume filter (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.2 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(sma_50_1w_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + weekly uptrend + volume
            if close[i] > highest_high[i] and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + weekly downtrend + volume
            elif close[i] < lowest_low[i] and not weekly_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle of Donchian channel
            mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle of Donchian channel
            mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals