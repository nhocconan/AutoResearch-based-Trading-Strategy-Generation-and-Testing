#!/usr/bin/env python3
"""
6h_ADX_Donchian_Breakout_12hTrend_Filter_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts in direction of 12h ADX>25 trend filter produce high-probability trades. ADX ensures we only trade in trending markets (avoiding chop), while Donchian breakouts capture momentum. Volume confirmation reduces false breakouts. Works in bull/bear via trend filter. Target: 50-120 trades over 4 years (12-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter (ADX) and Donchian levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nanmean(data[:period])
            # Rest: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = wilders_smooth(dx, 14)
    
    # Align ADX to 6h timeframe (completed 12h bars only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 12h Donchian(20) levels
    highest_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    highest_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_12h)
    lowest_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_12h)
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for ADX smoothing, 20 for Donchian/volume)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(highest_12h_aligned[i]) or
            np.isnan(lowest_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ADX trend filter (trending market)
        trending = adx_12h_aligned[i] > 25
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_12h_aligned[i]
        breakout_down = close[i] < lowest_12h_aligned[i]
        
        if trending and volume_spike and breakout_up:
            # Long signal: Donchian breakout up + trending + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif trending and volume_spike and breakout_down:
            # Short signal: Donchian breakout down + trending + volume spike
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Donchian_Breakout_12hTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0