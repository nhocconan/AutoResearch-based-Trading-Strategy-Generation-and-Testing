#!/usr/bin/env python3
# 4h_Pivot_Bounce_Trend_Volume
# Hypothesis: Price bouncing off weekly pivot S1/R1 with 1d EMA50 trend filter and volume spike confirmation on 4h timeframe.
# Uses weekly pivot levels from prior week to identify key support/resistance zones.
# In bull markets, long at S1 bounce; in bear markets, short at R1 rejection.
# Volume spike confirms institutional interest at these levels.
# EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws.
# Designed for low trade frequency (target: 20-40 trades/year) with high win rate.

name = "4h_Pivot_Bounce_Trend_Volume"
timeframe = "4h"
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

    # Get weekly data for pivot calculation (use 1w as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)

    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = 2 * pivot - weekly_high
    r1 = 2 * pivot - weekly_low
    
    # Align weekly pivot levels to 4h timeframe (hold until next week's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate volume spike threshold (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near S1 with bullish trend and volume spike
            if (low[i] <= s1_aligned[i] * 1.005 and  # Allow 0.5% tolerance for touch
                close[i] > s1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R1 with bearish trend and volume spike
            elif (high[i] >= r1_aligned[i] * 0.995 and  # Allow 0.5% tolerance for touch
                  close[i] < r1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or trend turns bearish
            if close[i] < s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or trend turns bullish
            if close[i] > r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals