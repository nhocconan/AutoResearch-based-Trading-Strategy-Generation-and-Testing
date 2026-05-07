#!/usr/bin/env python3
# 6H_WeeklyPivot_Breakout_DailyTrend_VolumeConfirm
# Hypothesis: Combines weekly pivot point breakout with daily EMA trend filter and volume confirmation. 
# Weekly pivots provide strong institutional support/resistance, daily EMA ensures trend alignment, 
# and volume confirms institutional participation. Designed for 6h timeframe to capture medium-term 
# breakouts with low trade frequency (~15-30 trades/year). Works in both bull and bear markets by 
# following the trend direction from higher timeframes.

name = "6H_WeeklyPivot_Breakout_DailyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2.0 * pivot_weekly - low_weekly
    s1_weekly = 2.0 * pivot_weekly - high_weekly
    
    # Align weekly pivot points to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Get daily EMA for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    # Volume spike: current volume > 1.5x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or 
            np.isnan(s1_weekly_aligned[i]) or np.isnan(ema20_daily_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume spike
            if (close[i] > r1_weekly_aligned[i] and 
                close[i] > ema20_daily_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume spike
            elif (close[i] < s1_weekly_aligned[i] and 
                  close[i] < ema20_daily_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below EMA or to S1 level (mean reversion)
            if close[i] < ema20_daily_aligned[i] or close[i] < s1_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA or to R1 level (mean reversion)
            if close[i] > ema20_daily_aligned[i] or close[i] > r1_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals