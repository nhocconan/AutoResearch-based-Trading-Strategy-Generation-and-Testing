#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Filter_v1
Hypothesis: On 1h timeframe, enter long when price breaks above daily Camarilla R3 with volume confirmation, enter short when price breaks below daily Camarilla S3 with volume confirmation. Use 4h ADX for trend strength filter (ADX > 25) to avoid whipsaws. Target 15-30 trades/year by using tight entry conditions and volume filter. Works in bull/bear by using volatility-based stops and mean-reversion exits when price returns to daily H3/L3 levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels (key reversal levels)
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    
    # Camarilla H3 and L3 levels (exit levels)
    h3 = close_1d + range_1d * 1.1 / 2
    l3 = close_1d - range_1d * 1.1 / 2
    
    # Align to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # === 4H ADX TREND STRENGTH ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range
    tr = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    tr[0] = high_4h[0] - low_4h[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
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
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # === VOLUME FILTER ===
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma  # Volume above 1.5x 20-period MA
    
    # === SESSION FILTER: 08-20 UTC ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions with volume and ADX trend filter
        long_breakout = (close[i] > r3_aligned[i]) and volume_filter[i] and (adx_aligned[i] > 25)
        short_breakout = (close[i] < s3_aligned[i]) and volume_filter[i] and (adx_aligned[i] > 25)
        
        # Exit conditions: return to H3/L3 levels or ADX weakens
        exit_long = (close[i] < h3_aligned[i]) or (close[i] > l3_aligned[i]) or (adx_aligned[i] < 20)
        exit_short = (close[i] > l3_aligned[i]) or (close[i] < h3_aligned[i]) or (adx_aligned[i] < 20)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals