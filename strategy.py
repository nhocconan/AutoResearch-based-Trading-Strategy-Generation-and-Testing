#!/usr/bin/env python3
# 1d_WeeklyPivot_Pullback_Trend_Filter
# Hypothesis: On daily timeframe, trade pullbacks to weekly pivot points (R1/S1) in direction of weekly trend.
# Uses weekly EMA50 as trend filter, weekly pivot levels (R1/S1) from prior week, and requires price to
# close back into the weekly range after touching the pivot level. Designed for low frequency (10-25 trades/year)
# to work in both bull and bear markets by fading extreme moves toward value.
# Maximum 2 positions at once to manage risk.

name = "1d_WeeklyPivot_Pullback_Trend_Filter"
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
    
    # Get weekly data for calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points (R1, S1, PP) from prior week
    # Using prior week's H, L, C to avoid look-ahead
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align all weekly indicators to daily timeframe
    ema_50_1w_d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    pp_1w_d = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_d = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily range for entry confirmation
    daily_range = high - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1w_d[i]) or np.isnan(r1_1w_d[i]) or 
            np.isnan(s1_1w_d[i]) or np.isnan(pp_1w_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or crosses R1 then pulls back into weekly range
            # and weekly trend is up (price > weekly EMA50)
            if (low[i] <= r1_1w_d[i] and  # Touched or went through R1
                close[i] > pp_1w_d[i] and  # Closed back above pivot point
                close[i] > ema_50_1w_d[i]):  # Above weekly trend
                signals[i] = 0.25
                position = 1
            # Short: Price touches or crosses S1 then pulls back into weekly range
            # and weekly trend is down (price < weekly EMA50)
            elif (high[i] >= s1_1w_d[i] and  # Touched or went through S1
                  close[i] < pp_1w_d[i] and  # Closed back below pivot point
                  close[i] < ema_50_1w_d[i]):  # Below weekly trend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly pivot or trend changes
            if close[i] < pp_1w_d[i] or close[i] < ema_50_1w_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly pivot or trend changes
            if close[i] > pp_1w_d[i] or close[i] > ema_50_1w_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals