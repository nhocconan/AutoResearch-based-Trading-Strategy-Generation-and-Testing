#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Price_Action_1dTrend
Hypothesis: Price reacting to weekly pivot points (S1/S2/R1/R2) with 1-day EMA trend filter and volume confirmation captures institutional reversal/continuation patterns. Works in bull (bounces off S1/S2 in uptrend) and bear (rejects at R1/R2 in downtrend). Low-frequency via 6h timeframe and confluence reduces overtrading.
"""
name = "6h_Weekly_Pivot_Price_Action_1dTrend"
timeframe = "6h"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, S2 = P-(H-L), R2 = P+(H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = 2 * pivot - weekly_high
    r1 = 2 * pivot - weekly_low
    s2 = pivot - (weekly_high - weekly_low)
    r2 = pivot + (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/bounces off S1 or S2 + 1d uptrend + volume
            if ((close[i] <= s1_aligned[i] * 1.002 and close[i] >= s1_aligned[i] * 0.998) or
                (close[i] <= s2_aligned[i] * 1.002 and close[i] >= s2_aligned[i] * 0.998)) and \
               close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches/rejects at R1 or R2 + 1d downtrend + volume
            elif ((close[i] >= r1_aligned[i] * 0.998 and close[i] <= r1_aligned[i] * 1.002) or
                  (close[i] >= r2_aligned[i] * 0.998 and close[i] <= r2_aligned[i] * 1.002)) and \
                  close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses pivot point in opposite direction
            if position == 1:
                if close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals