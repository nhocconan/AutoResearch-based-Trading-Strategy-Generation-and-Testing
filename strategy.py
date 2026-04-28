#!/usr/bin/env python3
"""
4h_Vortex_Trend_With_Adx_Threshold
Hypothesis: Uses Vortex Indicator (VI+) and VI- to determine trend direction, confirmed by ADX > 25 for trending markets. 
Trades only in strong trends to avoid whipsaws in ranging markets. Designed for low trade frequency (20-50/year) 
to minimize fee drag while capturing sustained directional moves. Works in both bull and bear by following 
the trend direction from Vortex crossover. Targets ~100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) 
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    di_plus = 100 * wilder_smooth(dm_plus, period) / atr
    di_minus = 100 * wilder_smooth(dm_minus, period) / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Vortex Indicator on 4h data
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[0], vm_plus])
    vm_minus = np.concatenate([[0], vm_minus])
    
    # True Range for Vortex
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr_vx = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_vx[0] = np.nan  # first value has no previous close
    
    # Smooth VM and TR
    def ema_smooth(arr, period):
        return pd.Series(arr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    vm_plus_smooth = ema_smooth(vm_plus, period)
    vm_minus_smooth = ema_smooth(vm_minus, period)
    tr_vx_smooth = ema_smooth(tr_vx, period)
    
    vi_plus = vm_plus_smooth / tr_vx_smooth
    vi_minus = vm_minus_smooth / tr_vx_smooth
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period*2)  # Ensure indicators are warm
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(vi_plus[i]) or 
            np.isnan(vi_minus[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Vortex crossover signals
        vi_cross_up = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_cross_down = vi_plus[i] < vi_minus[i] and vi_plus[i-1] >= vi_minus[i-1]
        
        # Exit when trend weakens or opposite crossover
        trend_weak = adx_aligned[i] < 20  # exit when trend weakens
        vi_reverse = vi_cross_down if position == 1 else vi_cross_up
        
        if vi_cross_up and strong_trend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif vi_cross_down and strong_trend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif trend_weak or vi_reverse:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Trend_With_Adx_Threshold"
timeframe = "4h"
leverage = 1.0