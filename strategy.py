#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ADX trend filter.
# Donchian breakouts capture trends, volume confirms institutional participation,
# and ADX ensures we only trade in trending markets (ADX > 25).
# This combination reduces false signals and works in both bull and bear markets by
# filtering for strong trends. Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day volume average (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    
    # 1-day ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        valid_start = ~np.isnan(arr)
        if not np.any(valid_start):
            return result
        first_valid = np.where(valid_start)[0][0]
        if first_valid + period >= len(arr):
            return result
        result[first_valid + period - 1] = np.nanmean(arr[first_valid:first_valid + period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(first_valid + period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 12h timeframe
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: current volume > 1.5x average 1d volume
    vol_confirm = volume > 1.5 * avg_vol_1d_aligned
    
    # ADX trend filter: ADX > 25 indicates strong trend
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    # Start from enough lookback for Donchian
    start_idx = max(lookback - 1, 20)  # Also need volume/ADX data
    
    for i in range(start_idx, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + volume confirmation + trend filter
            if (close[i] > highest_high[i] and 
                vol_confirm[i] and 
                trend_filter[i]):
                position = 1
                signals[i] = position_size
            # Short: break below lower Donchian + volume confirmation + trend filter
            elif (close[i] < lowest_low[i] and 
                  vol_confirm[i] and 
                  trend_filter[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian (stop and reverse)
            if close[i] < lowest_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian (stop and reverse)
            if close[i] > highest_high[i]:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0