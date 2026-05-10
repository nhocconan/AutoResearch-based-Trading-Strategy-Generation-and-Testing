#!/usr/bin/env python3
# 6H_WeeklyPivot_DailyTrend_VolumeBreakout
# Hypothesis: Breakout trades in direction of weekly pivot bias and daily trend with volume confirmation.
# Long when: price breaks above weekly R1 with daily uptrend and volume > 2x average.
# Short when: price breaks below weekly S1 with daily downtrend and volume > 2x average.
# Uses weekly pivot levels (calculated from prior week) as dynamic support/resistance.
# Works in bull/bear by following the higher timeframe trend and using volume to confirm institutional interest.
# Target: 15-30 trades/year per symbol.

name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
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
    
    # Weekly pivot levels (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate pivot points from previous week
    high_prev = df_1w['high'].values
    low_prev = df_1w['low'].values
    close_prev = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_prev[:-1] + low_prev[:-1] + close_prev[:-1]) / 3.0
    r1 = 2 * pivot - low_prev[:-1]
    s1 = 2 * pivot - high_prev[:-1]
    r2 = pivot + (high_prev[:-1] - low_prev[:-1])
    s2 = pivot - (high_prev[:-1] - low_prev[:-1])
    
    # Align weekly levels to 6h (no additional delay needed as pivot is known at week start)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot, additional_delay_bars=0)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1, additional_delay_bars=0)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1, additional_delay_bars=0)
    
    # Daily trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above weekly R1 with daily uptrend and volume confirmation
            if daily_up and volume_confirm and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 with daily downtrend and volume confirmation
            elif daily_down and volume_confirm and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price returns to pivot or trend weakens
            if close[i] < pivot_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price returns to pivot or trend weakens
            if close[i] > pivot_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals