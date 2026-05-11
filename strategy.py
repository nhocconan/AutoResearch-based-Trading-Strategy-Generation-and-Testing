#!/usr/bin/env python3
name = "1d_Donchian_Breakout_Volume_Filter"
timeframe = "1d"
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
    
    # Donchian breakout (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (SMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    weekly_uptrend = close > sma_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band, volume > 1.5x average, weekly uptrend
            if close[i] > highest_high[i] and volume[i] > 1.5 * volume_ma[i] and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band, volume > 1.5x average, weekly downtrend
            elif close[i] < lowest_low[i] and volume[i] > 1.5 * volume_ma[i] and not weekly_uptrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or weekly trend turns down
            if close[i] < lowest_low[i] or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or weekly trend turns up
            if close[i] > highest_high[i] or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals