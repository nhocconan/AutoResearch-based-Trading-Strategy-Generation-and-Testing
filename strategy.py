#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian upper (daily) AND weekly EMA20 rising AND volume > 1.5x 20-day average.
# Short when price breaks below 20-period Donchian lower (daily) AND weekly EMA20 falling AND volume > 1.5x 20-day average.
# Exit when price crosses back inside the Donchian channel.
# Donchian provides trend-following structure. Weekly EMA20 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian_20_1wEMA20_Volume"
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
    
    # 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels from daily OHLC
    # Using previous day's data to avoid look-ahead
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly EMA20 direction
    ema20_rising = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_rising[1:] = ema20_1w_aligned[1:] > ema20_1w_aligned[:-1]
    ema20_falling[1:] = ema20_1w_aligned[1:] < ema20_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-day average (using 1d data for consistency)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 2)  # Sufficient warmup for Donchian and EMA20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_rising[i]) or 
            np.isnan(ema20_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly EMA20 rising, volume filter
            long_cond = (close[i] > high_20_aligned[i]) and ema20_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, weekly EMA20 falling, volume filter
            short_cond = (close[i] < low_20_aligned[i]) and ema20_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals