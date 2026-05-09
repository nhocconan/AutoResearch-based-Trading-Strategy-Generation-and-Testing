#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_TrendFilter
# Hypothesis: Combines weekly pivot points (from 1w) as institutional S/R with Donchian(20) breakout from 6h for entry timing.
# Trend filter uses 60-period EMA on 6h to avoid counter-trend trades. Works in bull/bear: trend filter ensures we trade with higher timeframe momentum.
# Weekly pivots provide strong support/resistance; breakouts from these levels with trend alignment have higher follow-through.
# Targets 15-30 trades/year to minimize fee drag while capturing significant moves.

name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
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
    
    # Get weekly data for pivot points (using high, low, close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot levels to 6h timeframe (they update only when new weekly bar forms)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate 60-period EMA for trend filter on 6h
    ema_period = 60
    ema = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period - 1] = np.mean(close[0:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly pivot R1 AND above EMA (uptrend)
            if (close[i] > highest_high[i] and 
                close[i] > r1_aligned[i] and 
                close[i] > ema[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly pivot S1 AND below EMA (downtrend)
            elif (close[i] < lowest_low[i] and 
                  close[i] < s1_aligned[i] and 
                  close[i] < ema[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR below weekly pivot S1 OR below EMA (trend change)
            if (close[i] < lowest_low[i] or 
                close[i] < s1_aligned[i] or 
                close[i] < ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR above weekly pivot R1 OR above EMA (trend change)
            if (close[i] > highest_high[i] or 
                close[i] > r1_aligned[i] or 
                close[i] > ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals