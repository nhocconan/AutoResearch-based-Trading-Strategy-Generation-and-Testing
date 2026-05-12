#!/usr/bin/env python3
name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA150 for trend filter (using previous week's data to avoid look-ahead)
    ema150_1w = pd.Series(close_1w).ewm(span=150, adjust=False, min_periods=150).mean().values
    ema150_1w_aligned = align_htf_to_ltf(prices, df_1w, ema150_1w)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure Donchian has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema150_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly EMA150 + volume filter
            if close[i] > donchian_high[i] and close[i] > ema150_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly EMA150 + volume filter
            elif close[i] < donchian_low[i] and close[i] < ema150_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or below weekly EMA150
            if close[i] < donchian_low[i] or close[i] < ema150_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or above weekly EMA150
            if close[i] > donchian_high[i] or close[i] > ema150_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals