#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian channel breakout + volume confirmation + 1w EMA trend filter
# Uses weekly EMA to filter trend direction (avoid counter-trend trades) and daily Donchian for breakouts
# Volume surge confirms breakout strength. Designed for fewer trades (~20-40/year) to minimize fee drag.
# Works in bull (breakouts with trend) and bear (avoids counter-trend via weekly filter).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels (daily)
    highest_20 = np.full_like(high_1d, np.nan)
    lowest_20 = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        highest_20[i] = np.max(high_1d[i-19:i+1])
        lowest_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 50-period EMA (weekly) for trend filter
    alpha = 2.0 / (50 + 1)
    ema50_1w = np.full_like(close_1w, np.nan)
    ema50_1w[49] = np.mean(close_1w[:50])  # Simple average for seed
    for i in range(50, len(close_1w)):
        ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume ratio: current volume vs 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volume surge AND weekly uptrend
            if (close[i] > highest_20_aligned[i] and 
                volume_ratio > 3.0 and
                close[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 20-day low with volume surge AND weekly downtrend
            elif (close[i] < lowest_20_aligned[i] and 
                  volume_ratio > 3.0 and
                  close[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 20-day low or trend changes
            if (close[i] < lowest_20_aligned[i] or
                close[i] < ema50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 20-day high or trend changes
            if (close[i] > highest_20_aligned[i] or
                close[i] > ema50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_1w_EMA_Volume"
timeframe = "12h"
leverage = 1.0