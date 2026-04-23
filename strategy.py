#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Pivot + 1d Volume Spike + 6h EMA50 Trend Filter.
Long when price breaks above weekly R1 AND 6h EMA50 is rising AND 1d volume > 2.0x 20-period average.
Short when price breaks below weekly S1 AND 6h EMA50 is falling AND 1d volume > 2.0x 20-period average.
Exit when price touches opposite weekly pivot level (S1 for long, R1 for short) or EMA50 reverses.
Weekly pivots from 1w: R1 = 2*PP - low, S1 = 2*PP - high, where PP = (high+low+close)/3.
Uses 1w HTF for pivots and 1d HTF for volume to reduce noise. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 6h EMA50 for trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # Calculate 1w weekly pivots (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_pp = np.full(len(df_1w), np.nan)
    weekly_r1 = np.full(len(df_1w), np.nan)
    weekly_s1 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        pp = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        weekly_pp[i] = pp
        weekly_r1[i] = 2 * pp - low_1w[i]
        weekly_s1[i] = 2 * pp - high_1w[i]
    
    # Align weekly pivots to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 1d volume average for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above weekly R1 AND EMA50 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S1 AND EMA50 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 OR EMA50 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 OR EMA50 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyPivot_R1S1_Breakout_6hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0