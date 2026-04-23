#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
Target: 20-50 trades/year per symbol (80-200 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
Donchian channels provide robust breakout signals, ADX filters for trending markets only,
volume confirmation avoids false breakouts. Works in both bull/bear via ADX trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def ma_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    atr = ma_smoothing(tr, 14)
    dm_plus_smooth = ma_smoothing(dm_plus, 14)
    dm_minus_smooth = ma_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = ma_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel: max(high, 20)
    upper_20 = np.full_like(high_4h, np.nan, dtype=float)
    for i in range(len(high_4h)):
        if i >= 19:
            upper_20[i] = np.max(high_4h[i-19:i+1])
    
    # Lower channel: min(low, 20)
    lower_20 = np.full_like(low_4h, np.nan, dtype=float)
    for i in range(len(low_4h)):
        if i >= 19:
            lower_20[i] = np.min(low_4h[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need ADX, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_strong = adx_aligned[i] > 25
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND strong trend AND volume confirmation
            if close[i] > upper_20_aligned[i] and trend_strong and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND strong trend AND volume confirmation
            elif close[i] < lower_20_aligned[i] and trend_strong and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level or ADX weakening
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower
                if close[i] < lower_20_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper
                if close[i] > upper_20_aligned[i]:
                    exit_signal = True
            
            # Also exit if trend weakens (ADX < 20)
            if adx_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dADX_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0