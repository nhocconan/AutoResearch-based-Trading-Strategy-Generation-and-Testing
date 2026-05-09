#!/usr/bin/env python3
name = "12H_WeeklyPivot_R1_S1_Breakout_WeeklyTrend_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for pivot calculation (previous day's data for current day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly pivot levels from previous week
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Weekly Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 12.0
    weekly_s1 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 12.0
    
    # Align weekly pivot levels to 12h timeframe
    weekly_r1_12h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if weekly pivot levels not ready
        if np.isnan(weekly_r1_12h[i]) or np.isnan(weekly_s1_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R1 + volume confirmation + price above weekly EMA50
            if close[i] > weekly_r1_12h[i] and volume_confirm[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 + volume confirmation + price below weekly EMA50
            elif close[i] < weekly_s1_12h[i] and volume_confirm[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly S1
            if close[i] < weekly_s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly R1
            if close[i] > weekly_r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals