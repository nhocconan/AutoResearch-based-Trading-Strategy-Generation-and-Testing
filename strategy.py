#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w SMA trend filter + volume confirmation
# - Long when price breaks above 20-day high + price > 1w SMA50 + volume > 1.5x 20-day avg volume
# - Short when price breaks below 20-day low + price < 1w SMA50 + volume > 1.5x 20-day avg volume
# - Exit when price crosses opposite Donchian level (20-day low for longs, 20-day high for shorts)
# - Uses daily timeframe for trend following with weekly filter to avoid counter-trend trades
# - Volume confirmation ensures breakouts have institutional participation
# - Designed for 1d timeframe with selective entries to minimize trades and fee drag
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume confirmation
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period SMA on weekly timeframe
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if NaN in any indicator
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        volume = volume_1d[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        avg_volume = avg_volume_20[i]
        sma_50 = sma_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume > 1.5 * avg_volume
        
        if position == 0:
            # Long entry: price breaks above 20-day high + above weekly SMA50 + volume confirmation
            if price > donchian_high and price > sma_50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low + below weekly SMA50 + volume confirmation
            elif price < donchian_low and price < sma_50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day low (opposite Donchian level)
            if price < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day high (opposite Donchian level)
            if price > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wSMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0