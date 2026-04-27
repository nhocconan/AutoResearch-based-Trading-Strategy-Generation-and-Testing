#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ADX trend filter and volume confirmation.
# In trending markets (ADX > 25), price breaks beyond recent 20-period highs/lows with continuation.
# Uses 1d ADX for trend strength and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    period = 14
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smooth = smoothed_avg(tr, period)
    plus_dm_smooth = smoothed_avg(plus_dm, period)
    minus_dm_smooth = smoothed_avg(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels on 6h data (20-period)
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_channel, lower_channel = donchian_channels(high, low, 20)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(30, 20)  # ADX needs 30, Donchian needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long entry: price breaks above upper channel + strong trend + volume spike
            if (close[i] > upper_channel[i] and 
                strong_trend and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower channel + strong trend + volume spike
            elif (close[i] < lower_channel[i] and 
                  strong_trend and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower channel or trend weakens
            if (close[i] < lower_channel[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel or trend weakens
            if (close[i] > upper_channel[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_1dADX25_Volume_v1"
timeframe = "6h"
leverage = 1.0