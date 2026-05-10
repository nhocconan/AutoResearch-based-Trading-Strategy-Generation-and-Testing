#!/usr/bin/env python3
# 6H_1D_WeeklyPivot_Breakout_Trend
# Hypothesis: Use weekly pivot levels (from 1w) as structural support/resistance. Enter long when price breaks above weekly R1 in a bullish daily trend (close > EMA50), short when breaks below weekly S1 in bearish daily trend (close < EMA50). Uses weekly structure to filter noise and daily EMA for trend. Works in bull/bear by following daily trend direction. Target: 20-40 trades/year per symbol.

name = "6H_1D_WeeklyPivot_Breakout_Trend"
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
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily EMA50 for trend
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1w = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
    r1_1w = 2 * pivot_1w - low_1w[:-1]
    s1_1w = 2 * pivot_1w - high_1w[:-1]
    
    # Prepend NaN for alignment (current week's pivot uses prior week's data)
    pivot_1w = np.concatenate([[np.nan], pivot_1w])
    r1_1w = np.concatenate([[np.nan], r1_1w])
    s1_1w = np.concatenate([[np.nan], s1_1w])
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price breaks above weekly R1
            if bullish and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below weekly S1
            elif bearish and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price falls back below weekly pivot
            pivot_aligned = align_htf_to_ltf(prices, df_1w, np.concatenate([[np.nan], (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0]))
            if bearish or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price rises back above weekly pivot
            pivot_aligned = align_htf_to_ltf(prices, df_1w, np.concatenate([[np.nan], (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0]))
            if bullish or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals