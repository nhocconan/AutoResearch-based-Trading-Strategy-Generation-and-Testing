#!/usr/bin/env python3
# 1d_PivotReversal_WeeklyTrend_Volume
# Hypothesis: Daily reversals at weekly pivot support/resistance with weekly trend filter and volume confirmation.
# Uses weekly high/low/close to calculate pivot points (P), R1, S1. Enters long when price crosses above S1 with
# weekly uptrend (price > weekly EMA20) and volume > 1.5x 20-day average. Enters short when price crosses below R1
# with weekly downtrend (price < weekly EMA20) and volume confirmation. Exits when price crosses opposite pivot level.
# Designed for 1d timeframe to achieve 10-30 trades/year, suitable for both bull and bear markets by following
# weekly trend while using daily pivots for precise entries.

name = "1d_PivotReversal_WeeklyTrend_Volume"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Weekly data for pivot points, trend filter, and volume average
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly pivot points (based on previous week)
    def calculate_pivot(h, l, c):
        P = (h + l + c) / 3.0
        R1 = 2 * P - l
        S1 = 2 * P - h
        return P, R1, S1
    
    P = np.full_like(close_1w, np.nan)
    R1 = np.full_like(close_1w, np.nan)
    S1 = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        P[i], R1[i], S1[i] = calculate_pivot(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Weekly volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1w, 20)
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S1, above weekly EMA20, strong volume
            if close[i] > S1_aligned[i] and close[i] > ema_20_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1, below weekly EMA20, strong volume
            elif close[i] < R1_aligned[i] and close[i] < ema_20_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below P (pivot point) or below weekly EMA20
            if close[i] < P_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above P (pivot point) or above weekly EMA20
            if close[i] > P_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals