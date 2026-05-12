#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend_4hEMA21"
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
    
    # Load 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # 1h EMA21 for trend filter
    ema_21_1h = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_21_1h)
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_1h_aligned[i]) or 
            np.isnan(high_max[i]) or
            np.isnan(low_min[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1h EMA21 + volume filter
            if (close[i] > high_max[i] and 
                close[i] > ema_21_1h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 1h EMA21 + volume filter
            elif (close[i] < low_min[i] and 
                  close[i] < ema_21_1h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or below 1h EMA21
            if close[i] < low_min[i] or close[i] < ema_21_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or above 1h EMA21
            if close[i] > high_max[i] or close[i] > ema_21_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals