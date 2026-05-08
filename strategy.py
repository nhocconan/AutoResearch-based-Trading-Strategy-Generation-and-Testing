#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_1dPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Pivot points (previous day)
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    r1 = 2 * pivot - low_1d[:-1]
    s1 = 2 * pivot - high_1d[:-1]
    r2 = pivot + (high_1d[:-1] - low_1d[:-1])
    s2 = pivot - (high_1d[:-1] - low_1d[:-1])
    
    # Pad arrays to match length
    pivot = np.concatenate([np.array([np.nan]), pivot])
    r1 = np.concatenate([np.array([np.nan]), r1])
    s1 = np.concatenate([np.array([np.nan]), s1])
    r2 = np.concatenate([np.array([np.nan]), r2])
    s2 = np.concatenate([np.array([np.nan]), s2])
    
    # 1d trend direction: price above/below pivot
    trend_up = close_1d > pivot  # bullish bias
    trend_down = close_1d < pivot  # bearish bias
    
    # Align to 6h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + 1d bullish trend + volume
            if (close[i] > donchian_high[i] and
                trend_up_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down + 1d bearish trend + volume
            elif (close[i] < donchian_low[i] and
                  trend_down_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown or trend turns bearish
            if (close[i] < donchian_low[i] or
                not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up or trend turns bullish
            if (close[i] > donchian_high[i] or
                not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals