#!/usr/bin/env python3
name = "1d_Weekly_Donchian_Breakout_Volume_v2"
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
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period) - vectorized
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Use pandas rolling for efficiency
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=1).max().values
    donchian_low = low_series.rolling(window=20, min_periods=1).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema50
    
    # Align weekly indicators to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    # Volume moving average (20-period) for confirmation
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend changes
            if (close[i] < donchian_low_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend changes
            if (close[i] > donchian_high_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals