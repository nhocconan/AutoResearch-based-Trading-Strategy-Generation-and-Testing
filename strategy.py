#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Pivot + Daily EMA50 Trend + Volume Confirmation
Weekly pivot levels (from prior week) act as major support/resistance on 6h timeframe.
Breakout above weekly R1 or below weekly S1 with daily EMA50 trend alignment and volume
confirmation captures sustained momentum. Uses discrete sizing 0.25 to limit fee churn.
Timeframe 6h reduces noise and keeps trade frequency in optimal range (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot levels (R1, S1) from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close[0] = df_1w['close'].iloc[0]
    prev_week_high[0] = df_1w['high'].iloc[0]
    prev_week_low[0] = df_1w['low'].iloc[0]
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_range = prev_week_high - prev_week_low
    r1 = 2 * weekly_pivot - prev_week_low
    s1 = 2 * weekly_pivot - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 (breakout resistance) AND price > daily EMA50 (uptrend) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 (breakdown support) AND price < daily EMA50 (downtrend) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside prior week's pivot range OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < S1 (breakdown of support) OR price < daily EMA50
                if close[i] < s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R1 (breakout of resistance) OR price > daily EMA50
                if close[i] > r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyPivot_R1S1_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0