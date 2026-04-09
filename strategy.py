#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_v2
# Hypothesis: 6-hour breakouts at weekly pivot levels (S3/R3) with volume confirmation (>1.5x 20-bar average volume) and trend filter using 6h EMA(20).
# Weekly pivot levels act as strong support/resistance; breaks signal momentum continuation.
# EMA(20) filter ensures trades align with intermediate trend to reduce whipsaws in sideways markets.
# Volume filter reduces false breakouts. Works in bull markets (upward breaks above EMA) and bear markets (downward breaks below EMA).
# Target: 20-50 trades per year per symbol (~80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly close for pivot points (using prior week's data)
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly support/resistance levels
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Trend filter: 6h EMA(20)
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below S3 level
            if close[i] <= weekly_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above R3 level
            if close[i] >= weekly_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R3 with volume confirmation AND above EMA20 (uptrend)
            if close[i] > weekly_r3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5 and close[i] > ema_20[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S3 with volume confirmation AND below EMA20 (downtrend)
            elif close[i] < weekly_s3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5 and close[i] < ema_20[i]:
                position = -1
                signals[i] = -0.25
    
    return signals