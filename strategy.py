#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly EMA40 rising AND volume > 2x 20-day average.
# Short when price breaks below 20-day low AND weekly EMA40 falling AND volume > 2x 20-day average.
# Exit when price crosses back inside the 20-day range (between 20-day low and high).
# Daily timeframe provides sufficient signal frequency (target: 20-50 trades/year).
# Weekly EMA40 filters higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation and reduces false breakouts.
# Designed to work in both bull and bear markets via trend filter.

name = "1d_Donchian_20_WeeklyEMA40_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Weekly EMA40 direction
    ema40_rising = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_falling = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_rising[1:] = ema40_1w_aligned[1:] > ema40_1w_aligned[:-1]
    ema40_falling[1:] = ema40_1w_aligned[1:] < ema40_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Sufficient warmup for EMA40 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(ema40_rising[i]) or 
            np.isnan(ema40_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-day high, weekly EMA40 rising, volume filter
            long_cond = (close[i] > high_20[i]) and ema40_rising[i] and volume_filter[i]
            # Short conditions: price breaks below 20-day low, weekly EMA40 falling, volume filter
            short_cond = (close[i] < low_20[i]) and ema40_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 20-day low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above 20-day high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals