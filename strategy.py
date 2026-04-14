#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout + volume confirmation.
# Long when price breaks above 1d Donchian upper band AND 12h volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower band AND 12h volume > 1.5x 20-period average.
# Exit when price crosses back inside the 1d Donchian channel.
# Volume confirmation filters false breakouts. Designed to work in both bull and bear markets.
# Target: 15-35 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channel
    donchian_up = np.full_like(high_1d, np.nan)
    donchian_down = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        donchian_up[i] = np.max(high_1d[i-19:i+1])
        donchian_down[i] = np.min(low_1d[i-19:i+1])
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align indicators to 12h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_down_aligned = align_htf_to_ltf(prices, df_1d, donchian_down)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up_aligned[i]) or 
            np.isnan(donchian_down_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_ratio = volume_12h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: Donchian breakout + volume confirmation
            # Long: break above upper band AND volume > 1.5x average
            if (close[i] > donchian_up_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: break below lower band AND volume > 1.5x average
            elif (close[i] < donchian_down_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel
            if close[i] < donchian_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel
            if close[i] > donchian_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0