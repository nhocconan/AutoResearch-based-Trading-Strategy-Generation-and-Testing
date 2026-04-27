#!/usr/bin/env python3
"""
12h_WeeklyPivot_1dTrend_WithVolume
Hypothesis: Weekly pivot points (S1, R1) combined with daily EMA34 trend and volume confirmation.
Long when price breaks above weekly R1 in daily uptrend with volume spike (>1.5x avg volume).
Short when price breaks below weekly S1 in daily downtrend with volume spike.
Exit when price crosses daily EMA34 (trend reversal).
Designed for 12h timeframe to capture weekly structure with lower trade frequency.
Target: 15-30 trades/year to minimize fee drag in ranging/bear markets.
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (S1, R1) from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = np.full(len(close_1w), np.nan)
    weekly_r1 = np.full(len(close_1w), np.nan)
    weekly_s1 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):  # Start from 1 to use previous week
        weekly_pivot[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3
        weekly_r1[i] = 2 * weekly_pivot[i] - low_1w[i-1]
        weekly_s1[i] = 2 * weekly_pivot[i] - high_1w[i-1]
    
    # Align weekly pivot levels to 12h timeframe (previous week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate EMA(34) on daily close for trend filter
    ema_period = 34
    ema_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(df_1d['close'].iloc[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(df_1d)):
            ema_1d[i] = (df_1d['close'].iloc[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align EMA to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(2, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above weekly R1 in daily uptrend with volume
            if uptrend and volume_confirmation and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 in daily downtrend with volume
            elif downtrend and volume_confirmation and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below daily EMA34 (trend reversal)
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above daily EMA34 (trend reversal)
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_WeeklyPivot_1dTrend_WithVolume"
timeframe = "12h"
leverage = 1.0