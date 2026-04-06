#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, weekly MA(50) rising, volume > 2x average
# Enter short when: price breaks below Donchian(20) low, weekly MA(50) falling, volume > 2x average
# Exit when: price crosses opposite Donchian band OR weekly trend reverses
# Uses weekly trend to filter breakouts, targeting 50-150 trades over 4 years

name = "12h_donchian20_weekly_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: MA(50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ma_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ma_50_prev = np.roll(ma_50, 1)
    ma_50_prev[0] = ma_50[0]
    ma_50_rising = ma_50 > ma_50_prev
    ma_50_falling = ma_50 < ma_50_prev
    ma_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ma_50_rising)
    ma_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ma_50_falling)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ma_50_rising_aligned[i]) or np.isnan(ma_50_falling_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian low OR weekly trend turns falling
            if close[i] < low_min[i] or ma_50_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian high OR weekly trend turns rising
            if close[i] > high_max[i] or ma_50_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price breaks Donchian band + trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_max[i] and ma_50_rising_aligned[i]:
                    # Bullish breakout with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_min[i] and ma_50_falling_aligned[i]:
                    # Bearish breakout with falling weekly trend
                    signals[i] = -0.25
                    position = -1
    
    return signals