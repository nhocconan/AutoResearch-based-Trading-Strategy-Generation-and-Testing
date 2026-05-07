#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_Scalp_v1
# Hypothesis: Use weekly pivot points (R1/S1) for trend direction, combined with daily
#   EMA50 trend filter and volume confirmation on 6h. Long when price > weekly R1 and
#   above daily EMA50 with volume spike; short when price < weekly S1 and below daily
#   EMA50 with volume spike. Weekly pivots provide multi-week structure that works in
#   both bull and bear markets, while volume confirmation filters false breakouts.
#   Target: 15-25 trades/year to minimize fee drag.

name = "6h_WeeklyPivot_Trend_Scalp_v1"
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Get daily data for EMA50 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Align all indicators to 6h timeframe
    r1_weekly_6h = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_6h = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    ema_50_daily_6h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_weekly_6h[i]) or np.isnan(s1_weekly_6h[i]) or 
            np.isnan(ema_50_daily_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > weekly R1, above daily EMA50, volume spike
            if close[i] > r1_weekly_6h[i] and close[i] > ema_50_daily_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < weekly S1, below daily EMA50, volume spike
            elif close[i] < s1_weekly_6h[i] and close[i] < ema_50_daily_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price closes below weekly R1 or daily EMA50
            if close[i] < r1_weekly_6h[i] or close[i] < ema_50_daily_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above weekly S1 or daily EMA50
            if close[i] > s1_weekly_6h[i] or close[i] > ema_50_daily_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals