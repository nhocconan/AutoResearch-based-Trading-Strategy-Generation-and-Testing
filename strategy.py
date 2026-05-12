#!/usr/bin/env python3
# 6h Weekly Pivot + Volume Breakout with 1d Trend Filter
# Hypothesis: Weekly pivot levels (from 1d data) act as strong support/resistance.
# Price breaking above weekly R1 with volume confirmation and 1d uptrend = long signal.
# Price breaking below weekly S1 with volume confirmation and 1d downtrend = short signal.
# Weekly pivots are more significant than daily and work in both bull/bear markets
# by providing objective levels for breakout/breakdown trades.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_WeeklyPivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivots(daily_high, daily_low, daily_close):
    """Calculate weekly pivot points from daily OHLC data.
    Uses the prior week's data to calculate pivots for the current week.
    Standard formula: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    """
    n = len(daily_close)
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    # Need at least 5 days for a week
    if n < 5:
        return pivot, r1, s1
    
    # Calculate weekly aggregates from daily data
    # Week high = max of last 5 daily highs
    week_high = pd.Series(daily_high).rolling(window=5, min_periods=5).max().values
    # Week low = min of last 5 daily lows
    week_low = pd.Series(daily_low).rolling(window=5, min_periods=5).min().values
    # Week close = last daily close of the week (5th day back)
    week_close = pd.Series(daily_close).rolling(window=5, min_periods=5).apply(
        lambda x: x[-1] if len(x) == 5 else np.nan, raw=True
    ).values
    
    # Calculate pivots using prior week's data (shifted by 5 to avoid look-ahead)
    # We use the week that ended 5 days ago to calculate pivots for current week
    prior_week_high = np.roll(week_high, 5)
    prior_week_low = np.roll(week_low, 5)
    prior_week_close = np.roll(week_close, 5)
    
    # Only calculate where we have valid prior week data
    valid = ~(np.isnan(prior_week_high) | np.isnan(prior_week_low) | np.isnan(prior_week_close))
    
    if np.any(valid):
        p = (prior_week_high[valid] + prior_week_low[valid] + prior_week_close[valid]) / 3.0
        r = 2 * p - prior_week_low[valid]
        s = 2 * p - prior_week_high[valid]
        
        pivot[valid] = p
        r1[valid] = r
        s1[valid] = s
    
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for Weekly Pivots and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for weekly calculations
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily data
    weekly_pivot, weekly_r1, weekly_s1 = calculate_weekly_pivots(daily_high, daily_low, daily_close)
    
    # Align weekly pivot levels to 6h timeframe (using prior week's data, so already lagged)
    pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R1 with volume spike and 1d uptrend
            if (close[i] > r1_6h[i] and 
                vol_spike[i] and
                close[i] > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 with volume spike and 1d downtrend
            elif (close[i] < s1_6h[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls back below weekly pivot or trend weakens
            if close[i] < pivot_6h[i] or close[i] < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly pivot or trend weakens
            if close[i] > pivot_6h[i] or close[i] > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals