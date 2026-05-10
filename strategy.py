#!/usr/bin/env python3
# 12H_1D_1W_Camarilla_Breakout
# Hypothesis: Use weekly trend filter (price above/below weekly EMA50) and daily Camarilla pivot levels for entries.
# In uptrend, long when price breaks above daily R3; in downtrend, short when price breaks below daily S3.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal.
# Weekly trend reduces whipsaw, Camarilla provides precise entries/exits. Target: 20-30 trades/year.

name = "12H_1D_1W_Camarilla_Breakout"
timeframe = "12h"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla (shift by 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 2
    R2 = close_prev + (high_prev - low_prev) * 1.1 / 4
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 6
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 6
    S2 = close_prev - (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align all to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    weekly_uptrend_12h = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_12h = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(R1_12h[i]) or 
            np.isnan(S1_12h[i]) or np.isnan(S3_12h[i]) or
            np.isnan(weekly_uptrend_12h[i]) or np.isnan(weekly_downtrend_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above daily R3
            if weekly_uptrend_12h[i] > 0.5 and high[i] > R3_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below daily S3
            elif weekly_downtrend_12h[i] > 0.5 and low[i] < S3_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly downtrend OR price breaks below daily S1
            if weekly_downtrend_12h[i] > 0.5 or low[i] < S1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly uptrend OR price breaks above daily R1
            if weekly_uptrend_12h[i] > 0.5 or high[i] > R1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals