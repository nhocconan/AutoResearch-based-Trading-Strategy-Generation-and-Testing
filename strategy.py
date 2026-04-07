#!/usr/bin/env python3
"""
1h_volume_filtered_4h1d_trend_follow_v1
Hypothesis: On 1-hour timeframe, use 4h and 1d ADX for trend strength and volume surge for entry timing.
Long when: 4h ADX > 25, 1d ADX > 20 (trending market), and volume > 1.5x 20-period average.
Short when: 4h ADX > 25, 1d ADX > 20, and volume > 1.5x 20-period average (short on momentum).
Exit when 4h ADX < 20 (trend weakening) or volume drops below average.
Designed for 15-30 trades/year to minimize fee drift while capturing momentum in both bull and bear markets.
Volume filter reduces false signals and whipsaws, especially in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_filtered_4h1d_trend_follow_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for ADX (trend strength)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX for 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.concatenate([[np.nan], np.maximum(high_4h[1:] - high_4h[:-1], 0)])
    minus_dm = np.concatenate([[np.nan], np.maximum(low_4h[:-1] - low_4h[1:], 0)])
    
    # Fix where both are positive
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed values (Wilder's smoothing = alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(arr):
        smoothed = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                if i == 0:
                    smoothed[i] = np.nan
                else:
                    smoothed[i] = smoothed[i-1]
            else:
                if i == 0 or np.isnan(smoothed[i-1]):
                    smoothed[i] = arr[i]
                else:
                    smoothed[i] = smoothed[i-1] + alpha * (arr[i] - smoothed[i-1])
        return smoothed
    
    tr_smoothed = wilders_smoothing(tr)
    plus_dm_smoothed = wilders_smoothing(plus_dm)
    minus_dm_smoothed = wilders_smoothing(minus_dm)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = wilders_smoothing(dx)
    
    # Calculate ADX for 1d (same calculation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    plus_dm_1d = np.concatenate([[np.nan], np.maximum(high_1d[1:] - high_1d[:-1], 0)])
    minus_dm_1d = np.concatenate([[np.nan], np.maximum(low_1d[:-1] - low_1d[1:], 0)])
    
    plus_dm_1d = np.where((plus_dm_1d > minus_dm_1d) & (plus_dm_1d > 0), plus_dm_1d, 0)
    minus_dm_1d = np.where((minus_dm_1d > plus_dm_1d) & (minus_dm_1d > 0), minus_dm_1d, 0)
    
    tr_smoothed_1d = wilders_smoothing(tr_1d)
    plus_dm_smoothed_1d = wilders_smoothing(plus_dm_1d)
    minus_dm_smoothed_1d = wilders_smoothing(minus_dm_1d)
    
    plus_di_1d = 100 * plus_dm_smoothed_1d / tr_smoothed_1d
    minus_di_1d = 100 * minus_dm_smoothed_1d / tr_smoothed_1d
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d)
    
    # Align ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma  # Current volume / 20-period average
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after ADX warmup
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if ADX data not available
        if np.isnan(adx_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Skip if volume data not available
        if np.isnan(vol_ratio.iloc[i]):
            signals[i] = 0.0
            continue
            
        # Trend strength filter: both 4h and 1d ADX must indicate trending
        strong_trend = (adx_4h_aligned[i] > 25) and (adx_1d_aligned[i] > 20)
        weak_trend = adx_4h_aligned[i] < 20  # Exit trend when 4h ADX weakens
        
        # Volume surge filter: volume > 1.5x 20-period average
        volume_surge = vol_ratio.iloc[i] > 1.5
        
        if position == 1:  # Long position
            # Exit: trend weakens or volume drops
            if weak_trend or not volume_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend weakens or volume drops
            if weak_trend or not volume_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter on strong trend with volume surge
            if strong_trend and volume_surge:
                # Enter long on volume surge in uptrend
                signals[i] = 0.20
                position = 1
            # Note: We don't short on volume surge alone to avoid counter-trend shorts
            # Only short if we have a clear downtrend signal (could add later if needed)
    
    return signals