#!/usr/bin/env python3
"""
6h_WeeklyPivot_1dTrend_WithVolume_v2
Hypothesis: Weekly pivot breakout with daily trend filter and volume confirmation.
Long when price breaks above weekly R1 in daily uptrend with volume > 1.3x avg.
Short when price breaks below weekly S1 in daily downtrend with volume > 1.3x avg.
Exit on weekly pivot point touch or daily trend reversal.
Designed for 6h timeframe to capture multi-day swings with tight entries (target: 20-40/year).
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly high/low/close for pivot points (use previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_R1 = np.full(len(close_1w), np.nan)
    weekly_S1 = np.full(len(close_1w), np.nan)
    weekly_PP = np.full(len(close_1w), np.nan)  # Pivot Point
    
    for i in range(1, len(close_1w)):
        hl_range = high_1w[i-1] - low_1w[i-1]
        weekly_PP[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3
        weekly_R1[i] = close_1w[i-1] + (hl_range * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
        weekly_S1[i] = close_1w[i-1] - (hl_range * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate daily volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_period = 20
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(vol_ma_period, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-vol_ma_period:i+1])
    
    # Align all weekly and daily indicators to 6h timeframe
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    weekly_PP_aligned = align_htf_to_ltf(prices, df_1w, weekly_PP)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h volume confirmation (20-period average)
    vol_ma_6h_period = 20
    vol_ma_6h = np.full(n, np.nan)
    for i in range(vol_ma_6h_period, n):
        vol_ma_6h[i] = np.mean(volume[i-vol_ma_6h_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(1, 34, 20, 20)  # Weekly pivot needs 1 week, EMA(34), vol MA(20) both timeframes
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_R1_aligned[i]) or
            np.isnan(weekly_S1_aligned[i]) or
            np.isnan(weekly_PP_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h[i] if vol_ma_6h[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.3x average 6h volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: price breaks above weekly R1 in uptrend with volume
            if uptrend and volume_confirmation and price > weekly_R1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 in downtrend with volume
            elif downtrend and volume_confirmation and price < weekly_S1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches weekly PP or trend reverses
            if price <= weekly_PP_aligned[i] or price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price touches weekly PP or trend reverses
            if price >= weekly_PP_aligned[i] or price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_WeeklyPivot_1dTrend_WithVolume_v2"
timeframe = "6h"
leverage = 1.0