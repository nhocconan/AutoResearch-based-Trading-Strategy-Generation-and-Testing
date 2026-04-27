#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Camarilla pivot level breakouts (R1/S1) on 4h timeframe, filtered by 12h EMA50 trend and volume > 1.5x average.
Works in bull markets via breakout continuation and in bear via mean-reversion off extremes.
Target: 100-200 total trades over 4 years (~25-50/year) to avoid fee drag.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = np.zeros(len(close_4h))
    camarilla_s1 = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if high_4h[i] == low_4h[i]:
            camarilla_r1[i] = close_4h[i]
            camarilla_s1[i] = close_4h[i]
        else:
            camarilla_r1[i] = close_4h[i] + (high_4h[i] - low_4h[i]) * 1.1 / 12
            camarilla_s1[i] = close_4h[i] - (high_4h[i] - low_4h[i]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (already aligned, but ensure no look-ahead)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Align 12h EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Camarilla levels (from 4h), EMA (50), volume MA (20)
    start_idx = max(vol_ma_period, 1)  # Camarilla uses current 4h bar, EMA needs 50 periods
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 12h EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 in uptrend with volume
            if price > r1_aligned[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below S1 in downtrend with volume
            elif price < s1_aligned[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below S1 or trend reverses
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above R1 or trend reverses
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0