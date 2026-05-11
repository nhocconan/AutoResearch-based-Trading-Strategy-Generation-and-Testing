#!/usr/bin/env python3
# 1d_WeeklyPivot_DailyTrend_Filter
# Hypothesis: Use weekly pivot points (S1, R1) for entry on daily timeframe, filtered by weekly trend (EMA50) and volume confirmation.
# Weekly trend determines bias: price > weekly EMA50 = long bias, price < weekly EMA50 = short bias.
# Enter long when price crosses above weekly S1 with bullish weekly trend and volume > 1.5x 20-day average.
# Enter short when price crosses below weekly R1 with bearish weekly trend and volume > 1.5x 20-day average.
# Exit when price crosses the weekly pivot point (PP) or weekly trend changes.
# Designed to work in both bull and bear markets by using weekly trend filter and mean-reversion at weekly pivot levels.
# Target: Low trade frequency (<20/year) to minimize drag, high win rate via confluence.

name = "1d_WeeklyPivot_DailyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf  # Note: align_ltf_to_hlf is not standard; we use align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Weekly trend filter: EMA50 on weekly close
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_close > ema50_weekly  # True for uptrend
    
    # Align weekly data to daily timeframe (only use completed weekly bars)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above S1, weekly uptrend, volume confirmation
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and weekly_uptrend_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1, weekly downtrend, volume confirmation
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and not weekly_uptrend_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below PP OR weekly trend turns down
            if close[i] < pp_aligned[i] or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above PP OR weekly trend turns up
            if close[i] > pp_aligned[i] or weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals