#!/usr/bin/env python3
# 4h_12h_1D_Donchian_Breakout_VolumeTrend_Regime
# Hypothesis: On 4h timeframe, trade Donchian channel breakouts with volume confirmation and 12h trend filter.
# In trending markets (12h ADX > 25), trade breakout direction; in ranging markets (ADX < 25), trade breakouts with volume surge.
# Uses 1-day volume for confirmation to reduce noise. Targets 20-40 trades/year by requiring confluence.
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion at channel extremes.

name = "4h_12h_1D_Donchian_Breakout_VolumeTrend_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX for trend filter (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_12h = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr_12h
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = smooth_wilder(dx, 14)
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1d average volume for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (ADX > 25): trade breakout in direction of trend
            if adx_12h_aligned[i] > 25:
                # Long breakout above upper channel
                if high[i] > high_max[i-1]:
                    signals[i] = 0.30
                    position = 1
                # Short breakdown below lower channel
                elif low[i] < low_min[i-1]:
                    signals[i] = -0.30
                    position = -1
            # Ranging market (ADX < 25): trade breakouts with volume surge
            elif adx_12h_aligned[i] < 25:
                # Volume surge condition
                volume_surge = volume[i] > 2.0 * volume_ma_1d_aligned[i]
                # Long breakout with volume surge
                if high[i] > high_max[i-1] and volume_surge:
                    signals[i] = 0.30
                    position = 1
                # Short breakdown with volume surge
                elif low[i] < low_min[i-1] and volume_surge:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle of channel or trend reverses
            middle = (high_max[i-1] + low_min[i-1]) / 2
            if low[i] < middle or adx_12h_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price returns to middle of channel or trend reverses
            middle = (high_max[i-1] + low_min[i-1]) / 2
            if high[i] > middle or adx_12h_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals