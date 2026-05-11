#!/usr/bin/env python3
name = "6h_WeeklyPivot_BullBear_Switch_Trend_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels from 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    # Use previous week's close, high, low for pivot calculation
    weekly_close = df_1w['close'].shift(1).values  # previous week close
    weekly_high = df_1w['high'].shift(1).values    # previous week high
    weekly_low = df_1w['low'].shift(1).values      # previous week low
    weekly_close[0] = weekly_close[1] if len(weekly_close) > 1 else close[0]  # handle first value
    weekly_high[0] = weekly_high[1] if len(weekly_high) > 1 else high[0]
    weekly_low[0] = weekly_low[1] if len(weekly_low) > 1 else low[0]
    
    # Calculate weekly pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 12h trend filter using EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # ensure we have enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above weekly pivot AND above 12h EMA AND volume filter
            if close[i] > pivot_aligned[i] and close[i] > ema_12h_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly pivot AND below 12h EMA AND volume filter
            elif close[i] < pivot_aligned[i] and close[i] < ema_12h_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below weekly pivot OR below 12h EMA
            if close[i] < pivot_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above weekly pivot OR above 12h EMA
            if close[i] > pivot_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals