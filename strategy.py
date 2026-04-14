#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d KAMA trend for direction and 12h Donchian breakout for entry.
# KAMA adapts to market efficiency, reducing whipsaws in choppy markets.
# Donchian breakout from 12h provides entry signals with high probability of continuation.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Designed to work in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on 1d data
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=1))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / volatility[1:]
    er[0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian channels to 12h timeframe (no shift needed)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above 12h Donchian high or below 12h Donchian low
            # Only trade in direction of 1d KAMA (trend filter)
            
            # Long: price breaks above 12h Donchian high AND price > 1d KAMA
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > kama_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 12h Donchian low AND price < 1d KAMA
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < kama_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 12h Donchian low or price < 1d KAMA
            if (close[i] <= donchian_low_aligned[i] or 
                close[i] < kama_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 12h Donchian high or price > 1d KAMA
            if (close[i] >= donchian_high_aligned[i] or 
                close[i] > kama_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dKAMA_12hDonchian_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0