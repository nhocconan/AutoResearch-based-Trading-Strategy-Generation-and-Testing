#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses weekly pivot (from 1w data) to determine bias: long when price > weekly pivot, short when price < weekly pivot
# Breakout direction must align with weekly pivot bias to avoid counter-trend trades
# Volume confirmation (2.0x 24-bar MA) ensures institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Weekly pivot provides structural support/resistance that works in both bull and bear markets

name = "6h_Donchian20_1wPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian(20) for breakout levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling max/min for Donchian channels
    high_ma_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_ma_20)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_ma_20)
    
    # Calculate weekly pivot for trend filter (HTF)
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Volume confirmation: 2.0x 24-period average (~1 day for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 20  # Donchian20 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above weekly Donchian HIGH AND price > weekly pivot (bullish bias) AND volume spike
            if (close[i] > donchian_high_1w_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below weekly Donchian LOW AND price < weekly pivot (bearish bias) AND volume spike
            elif (close[i] < donchian_low_1w_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below weekly Donchian LOW OR price below weekly pivot (trend failure)
            if close[i] < donchian_low_1w_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly Donchian HIGH OR price above weekly pivot (trend failure)
            if close[i] > donchian_high_1w_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals