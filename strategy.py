#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_VolumeBreakout
Hypothesis: Price breaking above/below weekly pivot levels with daily trend alignment and volume confirmation captures institutional flow in both bull and bear markets. Weekly pivots act as key support/resistance; breakouts with volume indicate strong momentum. Daily trend filter ensures we trade with higher timeframe momentum, reducing whipsaws. Volume breakout filter ensures participation in genuine institutional moves. Designed for low-frequency, high-conviction trades on 6h timeframe.
"""
name = "6h_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    # Weekly support/resistance levels
    weekly_r1 = 2 * weekly_pivot - df_1w['low']
    weekly_s1 = 2 * weekly_pivot - df_1w['high']
    weekly_r2 = weekly_pivot + (weekly_r1 - weekly_s1)
    weekly_s2 = weekly_pivot - (weekly_r1 - weekly_s1)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2.values)
    s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2.values)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (institutional participation)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with daily uptrend and volume
            if close[i] > r1_6h[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with daily downtrend and volume
            elif close[i] < s1_6h[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to weekly pivot level (mean reversion at key level)
            if position == 1:
                if close[i] < pivot_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals