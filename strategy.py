#!/usr/bin/env python3
# 6h_Weekly_Pivot_D1_Trend_Filter
# Hypothesis: Combining weekly pivot points (from Monday open) with daily trend filter on 6h timeframe.
# Weekly pivot provides key institutional levels (PP, R1, S1, R2, S2) that act as support/resistance.
# Daily trend filter (EMA34) ensures we trade in the direction of higher timeframe momentum.
# Only take longs when price is above weekly PP and daily EMA34, shorts when below both.
# Entry on touch of weekly S1 (for longs) or R1 (for shorts) with confirmation from next bar.
# This structure should work in both bull and bear markets by adapting to weekly pivot levels.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag on 6h timeframe.

name = "6h_Weekly_Pivot_D1_Trend_Filter"
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
    
    # Daily trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Weekly pivot points (using Monday's OHLC)
    # We'll calculate weekly pivot using the first day of the week (Monday)
    # For simplicity, we use the weekly high/low/close from the 1d data
    # but we need to group by week. Instead, we use a rolling window of 5 days
    # as approximation for weekly data (5 trading days)
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly high, low, close (using last 5 days)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Calculate weekly pivot points
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot points to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or goes below S1 and closes back above it, with daily uptrend
            # We look for reversal from S1 support
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R1 and closes back below it, with daily downtrend
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S2 or daily trend turns down
            if (low[i] < s2_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R2 or daily trend turns up
            if (high[i] > r2_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals