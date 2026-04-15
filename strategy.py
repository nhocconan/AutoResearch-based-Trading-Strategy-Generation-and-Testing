#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Volume Spike and 1d ADX Trend Filter
# Buys when price breaks above 20-period high with above-average volume and strong trend (ADX>25)
# Sells when price breaks below 20-period low with above-average volume and strong trend (ADX>25)
# Uses 12h volume spike and 1d ADX to filter breakouts, reducing false signals in ranging markets.
# Designed to work in both bull and bear markets by trading breakouts in direction of trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 12h data for volume spike calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on daily timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume on 12h for spike detection
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = 1.5  # 1.5x average volume
    volume_12h_spike = pd.Series(volume_12h).rolling(window=20, min_periods=20).apply(
        lambda x: x[-1] > volume_spike_threshold * np.mean(x[:-1]) if len(x) >= 20 else False, raw=True
    ).values
    
    # Align volume spike signal to 4h timeframe
    volume_12h_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_12h_spike_aligned[i])):
            continue
        
        # Long entry: price breaks above 20-period high + volume spike + ADX > 25
        if (close[i] > highest_high[i] and
            volume_12h_spike_aligned[i] > 0.5 and  # True if volume spike occurred
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 20-period low + volume spike + ADX > 25
        elif (close[i] < lowest_low[i] and
              volume_12h_spike_aligned[i] > 0.5 and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < lowest_low[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADX"
timeframe = "4h"
leverage = 1.0