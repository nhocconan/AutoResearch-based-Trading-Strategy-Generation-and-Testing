#!/usr/bin/env python3
"""
4h_Adaptive_Range_Breakout_1DTrend
Hypothesis: On 4h timeframe, enter long when price breaks above 20-bar high with volume confirmation and 1d uptrend; enter short when price breaks below 20-bar low with volume confirmation and 1d downtrend. Uses ADX(14) to filter ranging markets (ADX<20) and avoid false breakouts. Position size 0.25 to limit drawdown. Designed to work in both bull and bear markets by following 1d trend direction.
"""

name = "4h_Adaptive_Range_Breakout_1DTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate ADX(14) on 1d for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha=1/14)
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h data for breakout and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Align 4h data to 4h index (no alignment needed as it's same timeframe)
    # Calculate 20-period rolling max/min for breakout levels
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Volume confirmation: 20-period average volume
    vol_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20-period high/low and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma20_4h[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50 direction
        uptrend_1d = close_4h[i] > ema50_1d_aligned[i]  # using 4h close vs 1d EMA
        downtrend_1d = close_4h[i] < ema50_1d_aligned[i]
        # ADX filter: only trade when trending (ADX >= 20)
        trending = adx_1d_aligned[i] >= 20
        
        # Breakout conditions
        breakout_high = high[i] > high_20[i-1]  # current high > previous 20-bar high
        breakout_low = low[i] < low_20[i-1]     # current low < previous 20-bar low
        
        # Volume confirmation: current volume > 1.5x 20-bar average volume
        volume_confirm = volume[i] > vol_ma20_4h[i] * 1.5
        
        if position == 0:
            # Long: breakout above 20-bar high in uptrend with volume and ADX
            if breakout_high and uptrend_1d and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: breakout below 20-bar low in downtrend with volume and ADX
            elif breakout_low and downtrend_1d and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 20-bar low or trend fails
            if low[i] < low_20[i-1] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 20-bar high or trend fails
            if high[i] > high_20[i-1] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals