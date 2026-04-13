#!/usr/bin/env python3
"""
6h_1d_Range_Breakout_with_Volume_and_ADX
Hypothesis: Uses daily range (high-low) to filter for low volatility conditions, then trades 6h breakouts with volume and ADX confirmation.
Works in bull markets by capturing breakout momentum, and in bear markets by avoiding false breakouts during low volatility.
ADX > 25 ensures we only trade when there's sufficient trend strength after breakout.
Target: 15-35 trades/year on 6h (60-140 total over 4 years).
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
    
    # Get daily data for range and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Calculate 50-period percentile of daily range for volatility filter (30th percentile = low volatility)
    daily_range_series = pd.Series(daily_range.values)
    daily_range_percentile = daily_range_series.rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Low volatility condition: daily range < 30th percentile
    low_volatility = daily_range_percentile < 30.0
    
    # Calculate ADX on daily (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = np.full_like(tr, np.nan, dtype=float)
    dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=float)
    
    # Initial values
    if len(tr) >= tr_period:
        atr[tr_period] = np.nanmean(tr[1:tr_period+1])
        dm_plus_smooth[tr_period] = np.nanmean(dm_plus[1:tr_period+1])
        dm_minus_smooth[tr_period] = np.nanmean(dm_minus[1:tr_period+1])
        
        # Wilder smoothing
        for i in range(tr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI+ and DI-
    di_plus = np.full_like(atr, np.nan, dtype=float)
    di_minus = np.full_like(atr, np.nan, dtype=float)
    dx = np.full_like(atr, np.nan, dtype=float)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    dx[valid] = (np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])) * 100
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan, dtype=float)
    adx_period = 14
    if len(dx) >= 2 * adx_period:
        adx[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
        for i in range(2*adx_period, len(dx)):
            if not np.isnan(dx[i-1]):
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Get 6h data for breakout levels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h Donchian channels (20-period)
    highest_20 = np.full_like(high_6h, np.nan, dtype=float)
    lowest_20 = np.full_like(low_6h, np.nan, dtype=float)
    
    for i in range(len(high_6h)):
        if i >= 19:
            highest_20[i] = np.max(high_6h[i-19:i+1])
            lowest_20[i] = np.min(low_6h[i-19:i+1])
    
    # Volume average for 6h
    vol_ma_20_6h = np.full_like(volume_6h, np.nan, dtype=float)
    for i in range(len(volume_6h)):
        if i >= 19:
            vol_ma_20_6h[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align all signals to main timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(low_volatility_aligned[i]) or \
           np.isnan(adx_aligned[i]) or \
           np.isnan(highest_20_aligned[i]) or \
           np.isnan(lowest_20_aligned[i]) or \
           np.isnan(vol_ma_20_6h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: low volatility + ADX > 25 + breakout with volume expansion
        if low_volatility_aligned[i] and adx_aligned[i] > 25:
            # Volume expansion on 6h
            volume_expansion = volume[i] > (vol_ma_20_6h_aligned[i] * 1.5) if i >= 20 else False
            
            # Long entry: price breaks above 6h Donchian high
            if high[i] > highest_20_aligned[i] and volume_expansion:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short entry: price breaks below 6h Donchian low
            elif low[i] < lowest_20_aligned[i] and volume_expansion:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: volatility increases or ADX weakens
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_Range_Breakout_with_Volume_and_ADX"
timeframe = "6h"
leverage = 1.0