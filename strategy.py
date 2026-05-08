#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band (20) AND 1w EMA20 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20) AND 1w EMA20 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (between upper and lower bands).
# This strategy captures breakouts with higher timeframe trend alignment and volume confirmation to avoid false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Donchian_20_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 1d data
    high_1d = high
    low_1d = low
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 1w EMA20 direction
    ema20_rising = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_rising[1:] = ema20_1w_aligned[1:] > ema20_1w_aligned[:-1]
    ema20_falling[1:] = ema20_1w_aligned[1:] < ema20_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Sufficient warmup for EMA20 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_rising[i]) or 
            np.isnan(ema20_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 1w EMA20 rising, volume filter
            long_cond = (close[i] > donchian_high[i]) and ema20_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, 1w EMA20 falling, volume filter
            short_cond = (close[i] < donchian_low[i]) and ema20_falling[i] and volume_filter[i]
            
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