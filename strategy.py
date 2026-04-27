#!/usr/bin/env python3
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
    
    # Get 1d data for trend filter and weekly data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data for ADX-based trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly ADX (14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip first NaN in tr
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(arr)):
            result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result
    
    tr_smoothed = wilder_smooth(tr, 14)
    dm_plus_smoothed = wilder_smooth(dm_plus, 14)
    dm_minus_smoothed = wilder_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, dm_plus_smoothed / tr_smoothed * 100, 0)
    di_minus = np.where(tr_smoothed != 0, dm_minus_smoothed / tr_smoothed * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = np.full_like(dx, np.nan)
    # First ADX is average of first 14 DX values
    if len(dx) >= 27:  # Need 14 for initial DX smoothing + 13 more
        adx[13] = np.nanmean(dx[1:14])  # Skip first NaN
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/4
    # S3 = C - (H - L) * 1.1/4
    pivot_1d = (high_1d + low_1d + close_1d_arr) / 3.0
    r3_1d = close_1d_arr + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = close_1d_arr - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align to 12h timeframe (previous day's levels available at open)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in uptrend (ADX > 25) with volume confirmation and during session
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in downtrend (ADX > 25) with volume confirmation and during session
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below EMA34 (trend change) OR ADX drops below 20 (trend weakening)
            if close[i] < ema34_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA34 (trend change) OR ADX drops below 20 (trend weakening)
            if close[i] > ema34_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_1wADX25_Volume_Session"
timeframe = "12h"
leverage = 1.0