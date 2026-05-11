# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Camarilla pivot breakout + volume confirmation + daily ADX trend filter.
This strategy targets medium-term breakouts from key intraday support/resistance levels (Camarilla levels)
which tend to work in both trending and ranging markets when combined with volume and trend filters.
Daily ADX ensures we only trade in trending conditions, avoiding choppy markets that cause whipsaws.
Volume confirmation ensures breakouts are supported by participation.
Designed for 12h to keep trade frequency low (target: 12-37 trades/year) to minimize fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily ADX for trend strength (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
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
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe (wait for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Daily Camarilla levels (using previous day's OHLC)
    # Typical Camarilla: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low), etc.
    # We'll use R3 and S3 as breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d)  # R3 level
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d)   # S3 level
    
    # Align Camarilla levels to 12h (use previous day's values)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = np.convolve(vol_1d, np.ones(20)/20, mode='same')  # simple moving average
    vol_ma20_1d[:19] = np.nan  # insufficient data for first 19 points
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for ADX and other indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        if adx_aligned[i] <= 25:
            # Weak trend - close any position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + volume filter
            if close[i] > camarilla_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + volume filter
            elif close[i] < camarilla_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or ADX drops below 20
            if close[i] < camarilla_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or ADX drops below 20
            if close[i] > camarilla_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals