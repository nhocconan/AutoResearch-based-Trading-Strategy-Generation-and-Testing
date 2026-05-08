#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when price crosses back below Donchian(10) high (for long) or above Donchian(10) low (for short).
# Uses Donchian channels for breakouts, ADX for trend strength, volume for confirmation.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost.

name = "4h_Donchian20_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian channels (10-period) for exit
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate ADX (14-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i]) or
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian(20) high, ADX > 25, volume spike
            long_cond = (close[i] > highest_high[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: break below Donchian(20) low, ADX > 25, volume spike
            short_cond = (close[i] < lowest_low[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian(10) high
            if close[i] < highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian(10) low
            if close[i] > lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals