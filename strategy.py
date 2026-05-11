#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 trend filter (requires 1-day close confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + above 1d EMA + volume spike
            if close[i] > high_max[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * volume_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + below 1d EMA + volume spike
            elif close[i] < low_min[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * volume_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below lower Donchian or below 1d EMA
            if close[i] < low_min[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above upper Donchian or above 1d EMA
            if close[i] > high_max[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals