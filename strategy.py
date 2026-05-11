#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_VolumeTrend"
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
    
    # Get daily data for 1-day Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Use pandas rolling with min_periods for proper calculation
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    daily_volume = df_1d['volume'].values
    vol_ma20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels and volume average to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for indicators
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > 20-day high + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < 20-day low + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < 20-day low (reverse signal) or volume drop
            if (close[i] < donchian_low_aligned[i] or 
                volume[i] < 0.8 * vol_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > 20-day high (reverse signal) or volume drop
            if (close[i] > donchian_high_aligned[i] or 
                volume[i] < 0.8 * vol_ma20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals