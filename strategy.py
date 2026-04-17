#!/usr/bin/env python3
"""
4h_RVOL_Breakout_Trend_v1
Relative Volume breakout with trend filter on 4h timeframe.
Buys when price breaks above 20-period high with 2x average volume and price above 50 EMA.
Sells when price breaks below 20-period low with 2x average volume and price below 50 EMA.
Uses 1d ADX > 20 to filter for trending markets only.
Designed to capture breakouts in both bull and bear markets with volume confirmation.
Target: 20-50 total trades over 4 years (5-12/year).
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
    
    # === 4h indicators ===
    # 20-period high/low for breakout
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # 50 EMA for trend filter
    ema_50 = np.full(n, np.nan)
    if n >= 50:
        ema_50[49] = np.mean(close[:50])
        for i in range(50, n):
            ema_50[i] = close[i] * 0.04 + ema_50[i-1] * 0.96  # alpha = 2/(50+1)
    
    # Volume average (20-period)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    # === 1d ADX for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM (14-period Wilder's smoothing)
    def wilders_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period:
            smoothed[period-1] = np.nansum(arr[1:period+1])
            for i in range(period, len(arr)):
                if not np.isnan(smoothed[i-1]) and not np.isnan(arr[i]):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(vol_avg_20[i]) or 
            np.isnan(volume[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long breakout: price > 20-period high AND volume > 2x average AND price > EMA50 AND ADX > 20
            if (close[i] > high_20[i] and 
                volume[i] > vol_avg_20[i] * 2 and 
                close[i] > ema_50[i] and 
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short breakout: price < 20-period low AND volume > 2x average AND price < EMA50 AND ADX > 20
            elif (close[i] < low_20[i] and 
                  volume[i] > vol_avg_20[i] * 2 and 
                  close[i] < ema_50[i] and 
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below 20-period low OR price < EMA50
            if (close[i] < low_20[i] or 
                close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high OR price > EMA50
            if (close[i] > high_20[i] or 
                close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RVOL_Breakout_Trend_v1"
timeframe = "4h"
leverage = 1.0