#!/usr/bin/env python3
"""
12h_donchian_breakout_volume_trend
Hypothesis: Combines 12h Donchian channel breakouts with volume confirmation and 1d ADX trend filter to capture strong trending moves. Works in bull markets (breakout continuation) and bear markets (breakdown continuation). Target: 15-25 trades/year to minimize fee drag. Uses both long and short positions for symmetry.
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
    
    # Get 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_channel = rolling_max(high_12h, 20)
    lower_channel = rolling_min(low_12h, 20)
    
    # Breakout signals
    breakout_up = high_12h > upper_channel
    breakout_down = low_12h < lower_channel
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    volume_expansion = volume_12h > (vol_ma_20 * 1.3)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    up_move = np.where(high_1d - np.roll(high_1d, 1) > 0, high_1d - np.roll(high_1d, 1), 0)
    down_move = np.where(np.roll(low_1d, 1) - low_1d > 0, np.roll(low_1d, 1) - low_1d, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_dm = wilders_smooth(up_move, period)
    minus_dm = wilders_smooth(down_move, period)
    
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    strong_trend = adx > 25
    
    # Align all signals to 12h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_12h, breakout_down.astype(float))
    volume_expansion_aligned = align_htf_to_ltf(prices, df_12h, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = breakout_up_aligned[i] > 0.5 and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        short_entry = breakout_down_aligned[i] > 0.5 and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        
        # Exit conditions: opposite breakout or loss of trend
        exit_long = position == 1 and (breakout_down_aligned[i] > 0.5 or strong_trend_aligned[i] <= 0.5)
        exit_short = position == -1 and (breakout_up_aligned[i] > 0.5 or strong_trend_aligned[i] <= 0.5)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            signals[i] = position * position_size
    
    return signals

name = "12h_donchian_breakout_volume_trend"
timeframe = "12h"
leverage = 1.0