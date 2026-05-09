#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Reversal_1dTrend
# Strategy: Fade at Camarilla R3/S3 levels from 1d timeframe with 1d trend filter
# Long when price touches S3 and 1d trend is up (price > 1d EMA50)
# Short when price touches R3 and 1d trend is down (price < 1d EMA50)
# Exit when price reaches opposite Camarilla level (R1/S1) or mean (Pivot)
# Uses mean reversion at extreme intraday levels with trend filter to avoid counter-trend trades
# Designed for 6h timeframe with selective entries to minimize trade frequency

name = "6h_Camarilla_R3S3_Reversal_1dTrend"
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
    
    # Calculate Camarilla levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels for each 1d bar: H, L, C from previous day
    # Camarilla formulas:
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low)
    # H1 = Close + 0.375*(High-Low)
    # L1 = Close - 0.375*(High-Low)
    # L2 = Close - 0.75*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # Pivot = (High + Low + Close)/3
    # We'll use R3 = H3 and S3 = L3 for fade
    
    # Shift to get previous day's values (lookback by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First bar has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    camarilla_H3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_L3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_H1 = prev_close_1d + 0.375 * (prev_high_1d - prev_low_1d)
    camarilla_L1 = prev_close_1d - 0.375 * (prev_high_1d - prev_low_1d)
    camarilla_P = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Align Camarilla levels to 6h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    H1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    L1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    P_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches or goes below S3 and 1d trend is up
            if low[i] <= L3_1d_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches or goes above R3 and 1d trend is down
            elif high[i] >= H3_1d_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches H1 or mean (Pivot)
            if high[i] >= H1_1d_aligned[i] or low[i] <= P_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches L1 or mean (Pivot)
            if low[i] <= L1_1d_aligned[i] or high[i] >= P_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals