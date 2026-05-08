#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian high (20 periods) AND 12h EMA25 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian low (20 periods) AND 12h EMA25 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel (between 20-period high and low).
# Donchian provides clear breakout levels, EMA25 filters intermediate trend, volume confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "4h_Donchian_20_12hEMA25_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Donchian channel (20-period high/low) - calculated on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h EMA25 for trend filter
    ema25_12h = pd.Series(df_12h['close']).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # 12h EMA25 direction
    ema25_rising = np.zeros_like(ema25_12h_aligned, dtype=bool)
    ema25_falling = np.zeros_like(ema25_12h_aligned, dtype=bool)
    ema25_rising[1:] = ema25_12h_aligned[1:] > ema25_12h_aligned[:-1]
    ema25_falling[1:] = ema25_12h_aligned[1:] < ema25_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(25, 20)  # Sufficient warmup for EMA25 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema25_12h_aligned[i]) or np.isnan(ema25_rising[i]) or 
            np.isnan(ema25_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 12h EMA25 rising, volume filter
            long_cond = (close[i] > donchian_high[i]) and ema25_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, 12h EMA25 falling, volume filter
            short_cond = (close[i] < donchian_low[i]) and ema25_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals