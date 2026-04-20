#!/usr/bin/env python3
# 6h_WeeklyPivot_RangeBreakout_VolumeFilter
# Hypothesis: In ranging markets (common in 2025-2026 BTC/ETH), price respects weekly pivot levels.
# Breakouts above weekly R1 or below S1 with volume confirmation capture new trends.
# Fade at R2/S2 in strong ranges. Weekly timeframe filters noise, volume ensures follow-through.
# Works in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_WeeklyPivot_RangeBreakout_VolumeFilter"
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
    
    # Get weekly data for pivot calculation (higher timeframe for stability)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots with one-week lag (use previous week's data)
    pivot = np.full_like(weekly_high, np.nan)
    r1 = np.full_like(weekly_high, np.nan)
    s1 = np.full_like(weekly_high, np.nan)
    r2 = np.full_like(weekly_high, np.nan)
    s2 = np.full_like(weekly_high, np.nan)
    
    for i in range(1, len(weekly_high)):  # Start from 1 to use prior week
        pivot[i] = (weekly_high[i-1] + weekly_low[i-1] + weekly_close[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - weekly_low[i-1]
        s1[i] = 2 * pivot[i] - weekly_high[i-1]
        r2[i] = pivot[i] + (weekly_high[i-1] - weekly_low[i-1])
        s2[i] = pivot[i] - (weekly_high[i-1] - weekly_low[i-1])
    
    # Align weekly pivots to 6h timeframe (already lagged by using prior week)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume filter: volume > 1.5x 20-period average to avoid breakouts on low volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Optional: fade at R2/S2 in ranging markets (counter-trend)
            elif close[i] > r2_aligned[i] and volume_filter[i]:
                # Fade at R2 - go short expecting reversion to pivot
                signals[i] = -0.15
                position = -1
            elif close[i] < s2_aligned[i] and volume_filter[i]:
                # Fade at S2 - go long expecting reversion to pivot
                signals[i] = 0.15
                position = 1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (failed breakout) or reaches R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (failed breakdown) or reaches S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals