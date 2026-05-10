#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: 4-hour breakouts from 12-hour Camarilla R1/S1 levels with 12-hour trend filter (EMA50) and volume confirmation.
# Uses 12h timeframe for trend direction to avoid counter-trend trades, and 12h Camarilla levels for precise entry/exit.
# Volume confirmation ensures breakout strength. Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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
    
    # 12h data for EMA50 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h Camarilla levels (based on previous 12h bar)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1 = np.full_like(close_12h, np.nan)
    S1 = np.full_like(close_12h, np.nan)
    for i in range(1, len(close_12h)):
        R1[i], S1[i] = calculate_camarilla(high_12h[i-1], low_12h[i-1], close_12h[i-1])
    
    # 12h volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_12h, 20)
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA50, strong volume
            if close[i] > R1_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 12h EMA50, strong volume
            elif close[i] < S1_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below 12h EMA50
            if close[i] < S1_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or above 12h EMA50
            if close[i] > R1_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals