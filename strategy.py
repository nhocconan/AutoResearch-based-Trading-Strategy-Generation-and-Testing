#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Breakouts from Camarilla R1/S1 levels (core support/resistance) with 4h EMA50 trend filter and volume confirmation.
# Uses 4h trend for direction, 1h for precise entry timing. Designed to avoid overtrading while capturing trends in both bull and bear markets.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # 1d data for Camarilla pivot calculation (standard formula)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12.0
    s1 = close_1d - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (24-period average for 1h timeframe = 1 day)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1, above 4h EMA50, strong volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1, below 4h EMA50, strong volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below Camarilla S1 or below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above Camarilla R1 or above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals