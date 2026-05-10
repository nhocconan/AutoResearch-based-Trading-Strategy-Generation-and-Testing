#!/usr/bin/env python3
# 6h_Camarilla_R4_S4_Breakout_1wTrend_Volume
# Hypothesis: 6-hour breakouts from weekly Camarilla R4/S4 levels with weekly trend filter (EMA50) and volume confirmation.
# Weekly EMA50 filters trend direction to avoid counter-trend trades; weekly Camarilla levels provide precise entry/exit;
# Volume confirmation ensures breakout strength. Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "6h"
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
    
    # Weekly data for EMA50 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Camarilla levels (based on previous week)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R4 = c + (range_ * 1.1000 / 2)  # R4 level
        S4 = c - (range_ * 1.1000 / 2)  # S4 level
        return R4, S4
    
    R4 = np.full_like(close_1w, np.nan)
    S4 = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        R4[i], S4[i] = calculate_camarilla(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Weekly volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1w, 20)
    
    # Align weekly indicators to 6h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4, above weekly EMA50, strong volume
            if close[i] > R4_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4, below weekly EMA50, strong volume
            elif close[i] < S4_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S4 or below weekly EMA50
            if close[i] < S4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R4 or above weekly EMA50
            if close[i] > R4_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals