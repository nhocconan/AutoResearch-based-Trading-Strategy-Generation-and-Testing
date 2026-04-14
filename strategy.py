#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot points for directional bias and 6h Donchian breakout for entry.
# Weekly pivot points provide institutional reference levels that work across market regimes.
# Long when price breaks above 6h Donchian upper band with price above weekly pivot.
# Short when price breaks below 6h Donchian lower band with price below weekly pivot.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Designed to work in both bull and bear markets by using weekly pivot as trend filter.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    pivot = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
    s1 = 2 * pivot - high_1w[:-1]
    r1 = 2 * pivot - low_1w[:-1]
    
    # Prepend NaN for alignment (no pivot for first week)
    pivot = np.concatenate([[np.nan], pivot])
    s1 = np.concatenate([[np.nan], s1])
    r1 = np.concatenate([[np.nan], r1])
    
    # Load 6h data ONCE for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    donchian_period = 20
    upper_donchian = pd.Series(high_6h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low_6h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_6h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_6h, lower_donchian)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(upper_donchian_aligned[i]) or
            np.isnan(lower_donchian_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts with weekly pivot filter
            # Long: price breaks above upper Donchian AND price above weekly pivot
            if (close[i] > upper_donchian_aligned[i] and 
                close[i] > pivot_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian AND price below weekly pivot
            elif (close[i] < lower_donchian_aligned[i] and 
                  close[i] < pivot_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian or breaks below weekly pivot
            if (close[i] <= lower_donchian_aligned[i] or 
                close[i] < pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to upper Donchian or breaks above weekly pivot
            if (close[i] >= upper_donchian_aligned[i] or 
                close[i] > pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_6hDonchian_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0