#!/usr/bin/env python3
name = "12h_Donchian_20_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d trend filter: EMA 34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Upper band: highest high of last 20 periods
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Volume filter: volume > 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Wait for EMA and Donchian to be ready
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + above 1d EMA + volume filter
            if close[i] > upper_12h_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + below 1d EMA + volume filter
            elif close[i] < lower_12h_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian band or below 1d EMA
            if close[i] < lower_12h_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian band or above 1d EMA
            if close[i] > upper_12h_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals