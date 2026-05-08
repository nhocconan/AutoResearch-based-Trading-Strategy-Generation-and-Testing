#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band (20) AND 1d EMA200 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20) AND 1d EMA200 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (between upper and lower bands).
# This strategy captures breakouts with long-term trend alignment and volume confirmation to avoid false breakouts.
# Donchian channels provide clear breakout levels. The 1d EMA200 filter ensures we trade with the long-term trend.
# Volume spike confirms institutional participation. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian_20_1dEMA200_Volume"
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
    
    # 1d data for Donchian calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d EMA200 direction
    ema200_rising = np.zeros_like(ema200_1d_aligned, dtype=bool)
    ema200_falling = np.zeros_like(ema200_1d_aligned, dtype=bool)
    ema200_rising[1:] = ema200_1d_aligned[1:] > ema200_1d_aligned[:-1]
    ema200_falling[1:] = ema200_1d_aligned[1:] < ema200_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200)  # Sufficient warmup for EMA200 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_rising[i]) or 
            np.isnan(ema200_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, 1d EMA200 rising, volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and ema200_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian low, 1d EMA200 falling, volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and ema200_falling[i] and volume_filter[i]
            
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