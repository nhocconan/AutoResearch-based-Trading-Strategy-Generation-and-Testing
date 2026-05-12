#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_VolumeBreakout"
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
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load daily data for volume average (20-day average)
    volume_1d = df_1d['volume'].values
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    # Calculate Donchian channels on 12h data (20 periods)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20d_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high + above daily EMA50 + volume > 1.5x 20-day average
            if high[i] > high_20[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.5 * vol_avg_20d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + below daily EMA50 + volume > 1.5x 20-day average
            elif low[i] < low_20[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.5 * vol_avg_20d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low or below daily EMA50
            if low[i] < low_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-period high or above daily EMA50
            if high[i] > high_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals