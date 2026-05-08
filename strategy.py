#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Long when price breaks above weekly Donchian upper band (20) AND 1w EMA20 rising AND volume > 1.5x 1d average.
# Short when price breaks below weekly Donchian lower band (20) AND 1w EMA20 falling AND volume > 1.5x 1d average.
# Exit when price crosses back inside weekly Donchian channel.
# This strategy captures breakouts with higher timeframe trend alignment and volume confirmation.
# Weekly timeframe reduces noise and improves signal quality for long-term trends.
# Target: 30-100 total trades over 4 years (7-25/year).

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
    
    # 1w data for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 1w EMA20 direction
    ema20_rising = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_rising[1:] = ema20_1w_aligned[1:] > ema20_1w_aligned[:-1]
    ema20_falling[1:] = ema20_1w_aligned[1:] < ema20_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average (1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup for EMA20 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_rising[i]) or 
            np.isnan(ema20_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 1w EMA20 rising, volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and ema20_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, 1w EMA20 falling, volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and ema20_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals