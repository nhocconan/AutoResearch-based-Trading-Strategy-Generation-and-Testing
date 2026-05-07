#!/usr/bin/env python3
# 1d_WeeklyPivot_RangeBreakout_WeekTrend_Filter
# Hypothesis: On 1d timeframe, buy when price breaks above weekly pivot R1 with weekly uptrend (EMA50),
# sell when breaks below S1 with weekly downtrend. Uses volume confirmation to avoid false breaks.
# Weekly trend filter reduces whipsaw in ranging markets. Target: 15-25 trades/year to minimize fee drag.
# Works in bull (breaks upward) and bear (breaks downward) via symmetric long/short logic.

name = "1d_WeeklyPivot_RangeBreakout_WeekTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (classic formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_1w_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(ema_50_1w_1d[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, above weekly EMA50 trend, volume confirmation
            if close[i] > r1_1d[i] and close[i] > ema_50_1w_1d[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, below weekly EMA50 trend, volume confirmation
            elif close[i] < s1_1d[i] and close[i] < ema_50_1w_1d[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or weekly trend turns down
            if close[i] < s1_1d[i] or close[i] < ema_50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or weekly trend turns up
            if close[i] > r1_1d[i] or close[i] > ema_50_1w_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals