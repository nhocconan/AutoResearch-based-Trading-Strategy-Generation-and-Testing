#!/usr/bin/env python3
# 1d_Weekly_Camarilla_Reversal
# Hypothesis: In both bull and bear markets, price reverts to weekly Camarilla pivot levels (S3, R3).
# Entries occur when price touches these levels on the daily timeframe with volume confirmation
# and weekly trend filter (price relative to weekly EMA200). Exits on opposite level touch.
# Designed for low trade frequency (10-20/year) to minimize fee drift.

name = "1d_Weekly_Camarilla_Reversal"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Weekly Camarilla levels: based on prior week's range
    # H = high, L = low, C = close of prior week
    H = high_1w
    L = low_1w
    C = close_1w
    RANGE = H - L
    # Camarilla formulas
    R3 = C + (RANGE * 1.1 / 2)
    S3 = C - (RANGE * 1.1 / 2)
    # Align to daily timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume confirmation (20-day average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # need weekly EMA200 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price touches or crosses below S3 with volume, above weekly EMA200
            if low[i] <= S3_aligned[i] and volume_confirm and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses above R3 with volume, below weekly EMA200
            elif high[i] >= R3_aligned[i] and volume_confirm and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses above R3
            if high[i] >= R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or crosses below S3
            if low[i] <= S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals