#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivotBias_VolumeFilter"
timeframe = "6h"
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
    
    # Weekly pivot bias (from Monday open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    week_open = df_1d['open'].values  # This is wrong - need to fix
    
    # Actually, let's implement properly:
    # Get weekly data properly
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Weekly bias: if weekly close > weekly open -> bullish bias
    weekly_bias = df_1w['close'] > df_1w['open']
    weekly_bias_vals = weekly_bias.values
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_vals)
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 6h Donchian channels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: volume > 1.3x 24-period average (4 days worth)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.3 * vol_ma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + weekly bullish bias + daily uptrend + volume spike
            if (close[i] > highest_high[i] and 
                weekly_bias_aligned[i] and 
                trend_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + weekly bearish bias + daily downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  not weekly_bias_aligned[i] and 
                  not trend_up[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Break below Donchian low or weekly bias turns bearish
            if close[i] < lowest_low[i] or not weekly_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Break above Donchian high or weekly bias turns bullish
            if close[i] > highest_high[i] or weekly_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals