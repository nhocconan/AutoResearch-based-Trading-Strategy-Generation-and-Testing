#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_v1
Hypothesis: On 12-hour timeframe, use weekly Camarilla pivot levels with 1-week trend filter for institutional-grade entries.
Long when price touches S3 level with weekly EMA(50) trending up, short when price touches R3 level with weekly EMA(50) trending down.
Exit when price reaches opposite pivot level (S1/R1). Uses weekly timeframe for trend to avoid whipsaw and align with institutional cycles.
Designed for 15-25 trades/year to minimize fee drift while capturing major reversals at key institutional levels.
Works in both bull/bear markets as Camarilla adapts to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (based on previous week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    
    s1 = weekly_close - (weekly_range * 1.1 / 12)
    s2 = weekly_close - (weekly_range * 1.1 / 6)
    s3 = weekly_close - (weekly_range * 1.1 / 4)
    r1 = weekly_close + (weekly_range * 1.1 / 12)
    r2 = weekly_close + (weekly_range * 1.1 / 6)
    r3 = weekly_close + (weekly_range * 1.1 / 4)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(weekly_ema_50), dtype=bool)
    weekly_trend_down = np.zeros(len(weekly_ema_50), dtype=bool)
    for i in range(1, len(weekly_ema_50)):
        if not np.isnan(weekly_ema_50[i]) and not np.isnan(weekly_ema_50[i-1]):
            weekly_trend_up[i] = weekly_ema_50[i] > weekly_ema_50[i-1]
            weekly_trend_down[i] = weekly_ema_50[i] < weekly_ema_50[i-1]
    
    # Align weekly data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float)) > 0.5
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float)) > 0.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(weekly_ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price reaches S1 level (profit target)
            if close[i] >= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level (profit target)
            if close[i] <= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with weekly trend alignment
            # Long: price touches S3 level with weekly uptrend
            if (close[i] <= s3_aligned[i] * 1.001 and close[i] >= s3_aligned[i] * 0.999 and 
                weekly_trend_up_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 level with weekly downtrend
            elif (close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001 and 
                  weekly_trend_down_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals