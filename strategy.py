#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) trend filter and volume spike confirmation.
- Uses 1d ADX > 25 to ensure trading only in trending markets (avoids chop/range)
- Donchian breakout from prior 20-period high/low on 12h timeframe
- Volume spike > 2.0x 20-period average to confirm institutional participation
- Position size: 0.25 discrete level to minimize fee churn
- Designed for 12-37 trades/year on 12h timeframe to avoid fee drag
- Works in both bull and bear markets by filtering for strong trends only
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Donchian, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        # Donchian breakout signals (using current close vs prior channels)
        breakout_up = close[i] > highest_20[i-1]  # Close above prior 20-period high
        breakout_down = close[i] < lowest_20[i-1]  # Close below prior 20-period low
        
        if position == 0:
            # Long: Donchian breakout up AND volume confirmation AND trend filter
            if breakout_up and volume_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND volume confirmation AND trend filter
            elif breakout_down and volume_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown (close below 20-period low)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout (close above 20-period high)
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dADX_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0