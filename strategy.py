#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_Trend_6hEMA50_v1
Hypothesis: 6h pullback to weekly pivot (PP) with 1w trend filter (price > weekly EMA50) and 6h EMA50 momentum confirmation. 
Targets 50-150 trades over 4 years by requiring confluence of weekly structure, trend alignment, and momentum. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via weekly trend filter and pullback logic.
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
    
    # Load weekly data ONCE before loop for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > weekly EMA50), -1 = downtrend (close < weekly EMA50)
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate weekly pivot points from previous week OHLC
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point (PP) = (H + L + C) / 3
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # 6h EMA50 for momentum confirmation
    ema_50_6h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 50 for 6h EMA)
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Pullback to weekly pivot with trend and momentum confirmation
        if position == 0:
            # Long: Price pulls back to weekly PP AND weekly uptrend AND 6h price > 6h EMA50
            if close[i] >= weekly_pp_aligned[i] * 0.998 and close[i] <= weekly_pp_aligned[i] * 1.002 and \
               trend_1w[i] == 1 and close[i] > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price pulls back to weekly PP AND weekly downtrend AND 6h price < 6h EMA50
            elif close[i] >= weekly_pp_aligned[i] * 0.998 and close[i] <= weekly_pp_aligned[i] * 1.002 and \
                 trend_1w[i] == -1 and close[i] < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price breaks above weekly PP * 1.02 OR weekly trend turns down
            if close[i] > weekly_pp_aligned[i] * 1.02 or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price breaks below weekly PP * 0.98 OR weekly trend turns up
            if close[i] < weekly_pp_aligned[i] * 0.98 or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Pullback_Trend_6hEMA50_v1"
timeframe = "6h"
leverage = 1.0