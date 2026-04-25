#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Trend_VolumeSpike
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot trend direction and volume confirmation.
Weekly pivot provides robust multi-week trend filter reducing whipsaws. Donchian breakouts capture momentum.
Only trade breakouts in direction of weekly trend to avoid counter-trend whipsaws. Volume spike confirms institutional interest.
Discrete sizing 0.25 to manage risk and minimize fee churn. Target: 12-30 trades/year (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points from previous week's OHLC
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    r2 = pivot_point + (prev_week_high - prev_week_low)
    s2 = pivot_point - (prev_week_high - prev_week_low)
    r3 = pivot_point + 2 * (prev_week_high - prev_week_low)
    s3 = pivot_point - 2 * (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and Donchian (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly trend bullish (close > EMA50) AND price > weekly R3 AND volume spike
            long_setup = (close[i] > donchian_high[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         (close[i] > r3_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Donchian low AND weekly trend bearish (close < EMA50) AND price < weekly S3 AND volume spike
            short_setup = (close[i] < donchian_low[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          (close[i] < s3_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bearish
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bullish
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0