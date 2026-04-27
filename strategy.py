#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud) with 1d ADX > 25 and volume > 1.5x average.
# Short when price breaks below Kumo with 1d ADX > 25 and volume > 1.5x average.
# Exit when price re-enters the Kumo.
# Uses Ichimoku for clear trend definition and breakout levels, ADX for trend strength,
# aiming for 12-37 trades per year on 6h timeframe.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    adx_period = 14
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = smooth_series(tr, adx_period)
    dm_plus_smoothed = smooth_series(dm_plus, adx_period)
    dm_minus_smoothed = smooth_series(dm_minus, adx_period)
    
    # Directional Indicators
    di_plus = np.full_like(tr_smoothed, np.nan)
    di_minus = np.full_like(tr_smoothed, np.nan)
    dx = np.full_like(tr_smoothed, np.nan)
    
    for i in range(len(tr_smoothed)):
        if not np.isnan(tr_smoothed[i]) and tr_smoothed[i] != 0:
            di_plus[i] = 100 * dm_plus_smoothed[i] / tr_smoothed[i]
            di_minus[i] = 100 * dm_minus_smoothed[i] / tr_smoothed[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < 2*adx_period-1:
            continue
        if np.isnan(dx[i]):
            adx[i] = np.nan
        elif np.isnan(adx[i-1]):
            adx[i] = np.nanmean(dx[adx_period-1:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(8, n):
        period9_high[i] = np.max(high[i-8:i+1])
        period9_low[i] = np.min(low[i-8:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(25, n):
        period26_high[i] = np.max(high[i-25:i+1])
        period26_low[i] = np.min(low[i-25:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(51, n):
        period52_high[i] = np.max(high[i-51:i+1])
        period52_low[i] = np.min(low[i-51:i+1])
    senkou_b = ((period52_high + period52_low) / 2)
    
    # For breakout detection, we need current cloud (Senkou Span A and B from 26 periods ago)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku (52), ADX, and volume MA20
    start_idx = max(52, 2*adx_period-1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        # Trend filter: require ADX > 25
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Kumo with trend and volume
            if price > kumo_top[i] and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Kumo with trend and volume
            elif price < kumo_bottom[i] and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Kumo (falls below Kumo top)
            if price < kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Kumo (rises above Kumo bottom)
            if price > kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Breakout_1dADX_Volume"
timeframe = "6h"
leverage = 1.0