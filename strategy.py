#!/usr/bin/env python3
"""
6h_1w_Pivot_Trend_Follow
Hypothesis: Weekly pivot points define strong support/resistance. In trending markets, price respects these levels. Uses weekly pivot direction (based on prior week's close) to filter 6h trend (EMA21) for breakout entries. Works in bull/bear by only trading with the weekly bias. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Pivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT POINTS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Weekly bias: bullish if prior week close > prior week pivot
    weekly_bullish = close_1w > pivot
    weekly_bearish = close_1w < pivot
    
    # Align to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    weekly_bullish_6h = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_6h = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # === 6h TREND (EMA21) ===
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly pivot-based trend filter: price > EMA21 in bullish week, < EMA21 in bearish week
    trend_bull = (ema_fast > pivot_6h) & weekly_bullish_6h.astype(bool)
    trend_bear = (ema_fast < pivot_6h) & weekly_bearish_6h.astype(bool)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(ema_fast[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: bullish week, price above EMA21, breaks above R1 with volume
        long_entry = trend_bull[i] and (close[i] > ema_fast[i]) and (close[i] > r1_6h[i]) and (vol_ratio[i] > 1.5)
        # Short: bearish week, price below EMA21, breaks below S1 with volume
        short_entry = trend_bear[i] and (close[i] < ema_fast[i]) and (close[i] < s1_6h[i]) and (vol_ratio[i] > 1.5)
        
        # Exit: trend reversal or price touches opposite S1/R1
        exit_long = position == 1 and (not trend_bull[i] or close[i] < s1_6h[i])
        exit_short = position == -1 and (not trend_bear[i] or close[i] > r1_6h[i])
        
        # Execute trades
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals