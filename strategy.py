#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_Volume_Trend
# Hypothesis: On 4h timeframe, trade Donchian channel breakouts with 1d trend filter and volume confirmation.
# In bull markets, breakouts above upper band with strong volume continue upward.
# In bear markets, breakouts below lower band with strong volume continue downward.
# Uses 1d ADX to filter trend strength and avoid whipsaws in ranging markets.
# Targets 20-40 trades/year by requiring strong volume (2x average) and ADX > 25.

name = "4h_1d_Donchian_Breakout_Volume_Trend"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
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
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            if adx_aligned[i] > 25:  # Only trade in trending markets
                # Long breakout above upper Donchian band with volume
                if (close[i] > highest_high[i] and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Short breakdown below lower Donchian band with volume
                elif (close[i] < lowest_low[i] and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: reverse when price crosses below midline or ADX weakens
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midline or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: reverse when price crosses above midline or ADX weakens
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midline or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals