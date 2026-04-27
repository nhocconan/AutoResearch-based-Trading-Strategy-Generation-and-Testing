#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and volume confirmation.
# Uses 4h Donchian channel breakouts for direction and 1h for precise entry timing.
# Includes session filter (08-20 UTC) to reduce noise and overtrading.
# Designed to work in both bull and bear markets by using breakouts with volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for higher timeframe context (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period) for trend
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high_4h_aligned[i]) or 
            np.isnan(lowest_low_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 4h upper Donchian + volume
        long_breakout = (close[i] > highest_high_4h_aligned[i-1] and volume_filter[i])
        # Short conditions: price breaks below 4h lower Donchian + volume
        short_breakout = (close[i] < lowest_low_4h_aligned[i-1] and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.20
            position = 1
        elif short_breakout:
            signals[i] = -0.20
            position = -1
        # Exit conditions: opposite 4h Donchian breakout
        elif position == 1 and close[i] < lowest_low_4h_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_4h_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_4hTrend_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0