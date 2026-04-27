#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy combining Donchian channel breakouts with volume confirmation and ADX trend filter.
# Uses 1-day ADX to filter for trending markets only, avoiding whipsaws in ranging conditions.
# Donchian(20) breakouts capture momentum; volume > 1.5x 20-period average confirms institutional participation.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets (upward breakouts) and bear markets (downward breakdowns) by only trading with the trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values are smoothed
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_sum = smoothed_avg(tr, 14)
    plus_dm_sum = smoothed_avg(plus_dm, 14)
    minus_dm_sum = smoothed_avg(minus_dm, 14)
    
    # DI values
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, 14)
    
    # Align ADX to 4h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20-period) on 4h data
    def donchian_channel(arr, period):
        upper = np.full_like(arr, np.nan)
        lower = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < period - 1:
                continue
            upper[i] = np.max(arr[i-period+1:i+1])
            lower[i] = np.min(arr[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channel(high, 20)
    # For lower band, we need to use low prices
    dc_lower = donchian_channel(low, 20)[1]
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            # In ranging markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Long breakout: price breaks above Donchian upper with volume
        if (close[i] > dc_upper[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below Donchian lower with volume
        elif (close[i] < dc_lower[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or volume drops
        elif position == 1 and (close[i] < dc_lower[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > dc_upper[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dADX25_VolumeFilter"
timeframe = "4h"
leverage = 1.0