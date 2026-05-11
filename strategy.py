#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 120-period EMA for 1d trend (long-term bias)
    ema120_1d = pd.Series(close_1d).ewm(span=120, adjust=False, min_periods=120).mean().values
    
    # 20-period average volume for volume spike filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 12h price
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    ema120_aligned = align_htf_to_ltf(prices, df_1d, ema120_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 120, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema120_aligned[i]) or np.isnan(vol_ma_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1d EMA120 + volume spike
            if close[i] > highest_high[i] and close[i] > ema120_aligned[i] and volume[i] > vol_ma_aligned[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 1d EMA120 + volume spike
            elif close[i] < lowest_low[i] and close[i] < ema120_aligned[i] and volume[i] > vol_ma_aligned[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals