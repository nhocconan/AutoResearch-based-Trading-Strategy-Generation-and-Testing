#!/usr/bin/env python3
"""
6h_12h_1d_Donchian_Volume_Trend_v1
Hypothesis: On 6h timeframe, buy Donchian(20) breakouts with 12h volume confirmation and 1d trend filter,
sell breakdowns with opposite conditions. Exit at opposite Donchian level. Uses 1d ADX trend strength
to filter choppy markets. Designed for low trade frequency (15-30/year) by requiring multiple confluence.
Works in bull/bear via 1d trend filter (ADX>25) and volatility-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Donchian_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H VOLUME AVERAGE (for confirmation) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_avg_12h = np.zeros_like(vol_12h)
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(vol_12h)):
        vol_sum += vol_12h[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= vol_12h[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg_12h[i] = vol_sum / vol_count
        else:
            vol_avg_12h[i] = 0.0
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    # === 1D ADX TREND STRENGTH ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nanmean(arr[1:period+1])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    tr_smooth = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(14, len(dx)):
        if i == 14:
            adx[i] = np.nanmean(dx[1:15])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend strength: ADX > 25
    trend_strong = adx > 25
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_strong_aligned = align_htf_to_ltf(prices, df_1d, trend_strong.astype(float))
    
    # === 6H DONCHIAN CHANNELS (20-period) ===
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_len-1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_avg_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(trend_strong_aligned[i]) or vol_avg_12h_aligned[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg_12h_aligned[i]
        
        # Only trade in strong trend regime (ADX > 25)
        in_trend = trend_strong_aligned[i] > 0.5
        
        # Entry conditions
        long_breakout = close[i] > upper[i]
        short_breakout = close[i] < lower[i]
        
        long_setup = long_breakout and vol_confirm and in_trend
        short_setup = short_breakout and vol_confirm and in_trend
        
        # Exit conditions: opposite Donchian level
        exit_long = close[i] < lower[i]
        exit_short = close[i] > upper[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals