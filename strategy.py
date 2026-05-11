#!/usr/bin/env python3
name = "12h_Donchian_20_Trend_Volume_Filter"
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
    
    # 1d trend: close above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 1w volume filter: volume > 1.5x 10-week average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    vol_1w = df_1w['volume'].values
    vol_ma10_1w = pd.Series(vol_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma10_1w)
    volume_filter = volume > 1.5 * vol_ma10_1w_aligned
    
    # Donchian(20) channels from previous 12h candles
    # Use rolling window on 12h data for upper/lower bands
    donchian_window = 20
    highest = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_window  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma10_1w_aligned[i]) or
            np.isnan(highest[i]) or np.isnan(lowest[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Donchian upper + 1d uptrend + volume filter
            if close[i] > highest[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian lower + 1d downtrend + volume filter
            elif close[i] < lowest[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Donchian lower or 1d trend down
            if close[i] < lowest[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Donchian upper or 1d trend up
            if close[i] > highest[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals