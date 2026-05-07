#!/usr/bin/env python3
# 12h_WMA_Cross_ADX_Trend
# Hypothesis: 12h strategy using weighted moving average cross (WMA10/30) with ADX trend filter and volume confirmation.
# Enters long when WMA10 crosses above WMA30, ADX > 25 (trending), and volume > 1.5x average.
# Enters short when WMA10 crosses below WMA30, ADX > 25 (trending), and volume > 1.5x average.
# Uses 1d timeframe for ADX calculation to avoid whipsaw in ranging markets.
# Designed for low trade frequency (~20-40 trades/year) to minimize fee drag while capturing trending moves.
# Works in both bull and bear markets by only trading when ADX confirms strong trend.

name = "12h_WMA_Cross_ADX_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wma(array, period):
    """Calculate Weighted Moving Average"""
    if len(array) < period:
        return np.full_like(array, np.nan, dtype=float)
    weights = np.arange(1, period + 1)
    return np.convolve(array, weights[::-1], mode='valid') / weights.sum()

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate WMA10 and WMA30 on 12h close prices
    wma10 = np.full(n, np.nan)
    wma30 = np.full(n, np.nan)
    
    # Calculate WMA10
    if n >= 10:
        wma10_vals = wma(close, 10)
        wma10[9:] = wma10_vals
    
    # Calculate WMA30
    if n >= 30:
        wma30_vals = wma(close, 30)
        wma30[29:] = wma30_vals
    
    # Calculate ADX components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (14-period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip first NaN
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection: 1.5x average volume (30-period for stability)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # WMA30 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(wma10[i]) or np.isnan(wma30[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # WMA cross signals
        wma10_prev = wma10[i-1] if i > 0 else np.nan
        wma30_prev = wma30[i-1] if i > 0 else np.nan
        wma10_cross_above = (wma10[i] > wma30[i]) and (wma10_prev <= wma30_prev)
        wma10_cross_below = (wma10[i] < wma30[i]) and (wma10_prev >= wma30_prev)
        
        if position == 0:
            # Long: WMA10 crosses above WMA30, ADX > 25 (trending), volume spike (>1.5x)
            if (wma10_cross_above and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: WMA10 crosses below WMA30, ADX > 25 (trending), volume spike (>1.5x)
            elif (wma10_cross_below and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: WMA10 crosses below WMA30 (trend reversal)
            if wma10_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WMA10 crosses above WMA30 (trend reversal)
            if wma10_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals