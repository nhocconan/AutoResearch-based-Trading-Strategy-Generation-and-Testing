#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_DailyTrend
# Hypothesis: On 6-hour chart, use weekly pivot points (from previous week) for breakout entries and daily EMA(34) for trend filter.
# Long when price breaks above weekly R1 with daily EMA(34) upward slope; short when price breaks below weekly S1 with daily EMA(34) downward slope.
# Exit when price returns to weekly pivot (PP) or trend reverses.
# Weekly pivots provide structural support/resistance; daily EMA filters for institutional trend alignment.
# Works in bull/bear: breaks capture momentum in trending markets, pivot exits prevent reversals in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WeeklyPivot_Breakout_DailyTrend"
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
    
    # Get weekly data for pivot points (previous week's H/L/C)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivots to 6h timeframe (use values from previous week)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Daily EMA slope (trend direction)
    ema_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_up = ema_slope > 0
    ema_down = ema_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_up[i]) or np.isnan(ema_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R1 AND daily EMA trending up
            if close[i] > r1_aligned[i] and ema_up[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 AND daily EMA trending down
            elif close[i] < s1_aligned[i] and ema_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly PP OR daily EMA trend turns down
            if close[i] < pp_aligned[i] or not ema_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly PP OR daily EMA trend turns up
            if close[i] > pp_aligned[i] or not ema_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals