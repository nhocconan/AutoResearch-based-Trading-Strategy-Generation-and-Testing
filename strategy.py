#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d EMA50 trend filter
# Donchian breakouts capture momentum in trending markets. Volume confirmation filters weak breakouts.
# EMA50 from 1d provides higher timeframe trend bias to avoid counter-trend trades.
# Works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries
name = "4h_Donchian20_12hVolume_1dEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h_avg = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_12h_avg_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_avg)
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_12h_avg_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1d EMA50 + volume above 12h average
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > volume_12h_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 1d EMA50 + volume above 12h average
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > volume_12h_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or below 1d EMA50
            if (close[i] < low_20[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or above 1d EMA50
            if (close[i] > high_20[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals