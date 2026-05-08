#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d ADX Trend Filter + Volume Confirmation
# Uses daily ADX to identify trending markets, Donchian(20) breakout on 4h for entry,
# and volume > 1.5x 20-period average for confirmation. Works in both bull and bear
# markets by only taking breakouts in the direction of the daily trend (ADX > 25).
# Target: 25-40 trades/year to stay under fee drag limits.

name = "4h_Donchian20_1dADX25_VolumeConfirm"
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
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_daily[1:] - high_daily[:-1]
    down_move = low_daily[:-1] - low_daily[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
            else:
                smoothed[i] = np.nansum(arr[i-period+1:i+1])
        return smoothed
    
    atr_14 = smooth_wilder(tr, 14)
    plus_dm_14 = smooth_wilder(plus_dm, 14)
    minus_dm_14 = smooth_wilder(minus_dm, 14)
    
    # DI values
    plus_di_14 = np.full_like(atr_14, np.nan)
    minus_di_14 = np.full_like(atr_14, np.nan)
    for i in range(len(atr_14)):
        if not np.isnan(atr_14[i]) and atr_14[i] != 0:
            plus_di_14[i] = 100 * plus_dm_14[i] / atr_14[i]
            minus_di_14[i] = 100 * minus_dm_14[i] / atr_14[i]
    
    # DX and ADX
    dx = np.full_like(atr_14, np.nan)
    for i in range(len(atr_14)):
        if not np.isnan(plus_di_14[i]) and not np.isnan(minus_di_14[i]):
            di_sum = plus_di_14[i] + minus_di_14[i]
            if di_sum != 0:
                dx[i] = 100 * np.abs(plus_di_14[i] - minus_di_14[i]) / di_sum
    
    adx_14 = smooth_wilder(dx, 14)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_high_20[i] = np.max(high[i-20:i+1])
        lowest_low_20[i] = np.min(low[i-20:i+1])
    
    # Calculate volume average (20-period)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_daily, adx_14)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in trending market (ADX > 25) with volume confirmation
            trending_market = adx_14_aligned[i] > 25
            
            # Long when price breaks above upper Donchian band
            long_condition = (
                close[i] > highest_high_20[i] and   # breakout above upper band
                trending_market and                 # trending market (ADX > 25)
                vol_confirm                         # volume confirmation
            )
            
            # Short when price breaks below lower Donchian band
            short_condition = (
                close[i] < lowest_low_20[i] and     # breakout below lower band
                trending_market and                 # trending market (ADX > 25)
                vol_confirm                         # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below midpoint of Donchian channel or ADX drops
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above midpoint of Donchian channel or ADX drops
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals