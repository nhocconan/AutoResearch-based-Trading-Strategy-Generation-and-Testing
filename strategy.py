#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend_v1
Hypothesis: On daily timeframe, enter long when price breaks above weekly Camarilla R3 with strong trend confirmation (ADX>25), enter short when price breaks below weekly Camarilla S3 with ADX>25. Uses weekly Camarilla levels for structure and ADX for trend strength. Designed for very few trades (target 7-25/year) to avoid fee drag and work in both bull and bear markets by requiring strong trend conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY INDICATORS: Camarilla pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla R3 and S3 levels (key reversal levels)
    r3 = close_1w + range_1w * 1.1 / 4
    s3 = close_1w - range_1w * 1.1 / 4
    
    # Align to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === WEEKLY ADX TREND STRENGTH ===
    # Calculate True Range
    tr = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr[0] = high_1w[0] - low_1w[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr14 = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr14 > 0, dm_plus_smooth / atr14 * 100, 0)
    di_minus = np.where(atr14 > 0, dm_minus_smooth / atr14 * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with ADX trend filter
        long_breakout = (close[i] > r3_aligned[i]) and (adx_aligned[i] > 25)
        short_breakout = (close[i] < s3_aligned[i]) and (adx_aligned[i] > 25)
        
        # Exit conditions: reversal back inside Camarilla H3-L3 range or ADX weakens
        h3 = close_1w + range_1w * 1.1 / 2
        l3 = close_1w - range_1w * 1.1 / 2
        h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
        l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
        
        exit_long = (close[i] < h3_aligned[i]) or (close[i] > l3_aligned[i]) or (adx_aligned[i] < 20)
        exit_short = (close[i] > l3_aligned[i]) or (close[i] < h3_aligned[i]) or (adx_aligned[i] < 20)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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