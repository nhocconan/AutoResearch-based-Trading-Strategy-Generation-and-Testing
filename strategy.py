#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 20-period high AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period low AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# This strategy targets breakouts with trend strength confirmation and avoids false signals.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for trending conditions only.

name = "6h_Donchian_20_1dADX25_Volume"
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
    
    # Daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 6h data
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX on daily data
    # ADX requires +DI, -DI, and DX calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[1:period]) if not np.all(np.isnan(values[1:period])) else np.nan
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smoothed = wilders_smoothing(tr, 25)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 25)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 25)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 25)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Sufficient warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high, ADX > 25, volume filter
            long_cond = (close[i] > high_max20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: price breaks below 20-period low, ADX > 25, volume filter
            short_cond = (close[i] < low_min20[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 20-period low
            if close[i] < low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above 20-period high
            if close[i] > high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals