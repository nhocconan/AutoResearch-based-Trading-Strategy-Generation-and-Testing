#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Breakouts from Camarilla R3/S3 levels with 12h EMA21 trend filter and volume confirmation.
# Camarilla R3/S3 provide strong support/resistance levels. Combined with 12h trend filter and volume
# confirmation, this strategy aims to capture meaningful breakouts in both bull and bear markets.
# Designed for 4h to balance trade frequency and signal quality, targeting ~25-40 trades/year.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (standard formula)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 4.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA21 for trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume confirmation (20-period average for 4h timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, above 12h EMA21, strong volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema_21_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, below 12h EMA21, strong volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema_21_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Camarilla S3 or below 12h EMA21
            if close[i] < s3_aligned[i] or close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Camarilla R3 or above 12h EMA21
            if close[i] > r3_aligned[i] or close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals