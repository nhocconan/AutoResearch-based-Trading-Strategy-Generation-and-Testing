#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 1w pivot levels for trend direction and 6h Donchian channels for entry/exit
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1w pivot trend and using Donchian for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "6h_Donchian20_1wPivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 1w pivot point (using completed weekly bars)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point for trend direction
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align 1w pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Donchian breakout with 1w pivot trend filter
        # Long: Price breaks above 1d Donchian upper + volume spike + price above 1w pivot (uptrend bias)
        # Short: Price breaks below 1d Donchian lower + volume spike + price below 1w pivot (downtrend bias)
        if position == 0:
            if (close[i] > upper_20_aligned[i] and volume_spike and 
                close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lower_20_aligned[i] and volume_spike and 
                  close[i] < pivot_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below 1d Donchian lower OR price below 1w pivot (trend change)
            if close[i] < lower_20_aligned[i] or close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above 1d Donchian upper OR price above 1w pivot (trend change)
            if close[i] > upper_20_aligned[i] or close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals