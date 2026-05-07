#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_Filter_Volume_Signal
# Hypothesis: Use weekly pivot points as structural levels and daily trend filter with volume confirmation on 6h timeframe.
# Weekly pivots provide strong support/resistance that work in both bull and bear markets. 
# Price above/below weekly pivot determines bias, with entries on retests of pivot levels with volume confirmation.
# Daily EMA50 filter ensures we only trade in the direction of higher timeframe trend.
# Target: 20-40 trades/year per symbol to stay under 300 total trades limit.

name = "6h_WeeklyPivot_Trend_Filter_Volume_Signal"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    open_w = df_w['open'].values
    
    # Weekly pivot point calculation
    pw = (high_w + low_w + close_w) / 3.0
    r1w = 2 * pw - low_w
    s1w = 2 * pw - high_w
    r2w = pw + (high_w - low_w)
    s2w = pw - (high_w - low_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) == 0:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all levels to 6h timeframe
    pw_aligned = align_htf_to_ltf(prices, df_w, pw)
    r1w_aligned = align_htf_to_ltf(prices, df_w, r1w)
    s1w_aligned = align_htf_to_ltf(prices, df_w, s1w)
    r2w_aligned = align_htf_to_ltf(prices, df_w, r2w)
    s2w_aligned = align_htf_to_ltf(prices, df_w, s2w)
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume spike detection: 1.8x average volume (30-period for stability)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure we have volume MA and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pw_aligned[i]) or np.isnan(r1w_aligned[i]) or np.isnan(s1w_aligned[i]) or
            np.isnan(r2w_aligned[i]) or np.isnan(s2w_aligned[i]) or np.isnan(ema50_d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly pivot (bullish bias), retesting S1 or S2 with volume confirmation
            if (close[i] > pw_aligned[i] and 
                (abs(close[i] - s1w_aligned[i]) < 0.005 * close[i] or abs(close[i] - s2w_aligned[i]) < 0.005 * close[i]) and
                close[i] > ema50_d_aligned[i] and
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot (bearish bias), retesting R1 or R2 with volume confirmation
            elif (close[i] < pw_aligned[i] and 
                  (abs(close[i] - r1w_aligned[i]) < 0.005 * close[i] or abs(close[i] - r2w_aligned[i]) < 0.005 * close[i]) and
                  close[i] < ema50_d_aligned[i] and
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below weekly pivot or reaches R1
            if close[i] < pw_aligned[i] or close[i] > r1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above weekly pivot or reaches S1
            if close[i] > pw_aligned[i] or close[i] < s1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals