#!/usr/bin/env python3
# 6h_Weekly_Pivot_Plus_Daily_Momentum
# Hypothesis: 6-hour breakouts from weekly CPR (Central Pivot Range) with daily momentum filter.
# Weekly CPR provides institutional support/resistance from weekly structure. Daily ROC(10) > 0 filters for bullish momentum, ROC(10) < 0 for bearish.
# Volume confirmation ensures breakout strength. Designed for 6h to achieve 12-37 trades/year, working in both bull and bear markets by following weekly structure.

name = "6h_Weekly_Pivot_Plus_Daily_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for CPR calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly CPR (Central Pivot Range)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    bc = (high_1w + low_1w) / 2.0  # Balance Point
    tc = (pivot * 2) - bc          # Top Central
    bc = (pivot * 2) - tc          # Bottom Central (recalculate)
    
    tc = np.where(tc >= bc, tc, bc)  # Ensure TC >= BC
    bc = np.where(bc <= tc, bc, tc)  # Ensure BC <= TC
    
    # Daily ROC(10) for momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    roc_10 = np.full_like(close_1d, np.nan)
    for i in range(10, len(close_1d)):
        roc_10[i] = (close_1d[i] - close_1d[i-10]) / close_1d[i-10] * 100
    
    # Daily volume confirmation: 20-period average
    volume_1d = df_1d['volume'].values
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align weekly indicators to 6h timeframe (wait for weekly bar to close)
    tc_aligned = align_htf_to_ltf(prices, df_1w, tc)
    bc_aligned = align_htf_to_ltf(prices, df_1w, bc)
    
    # Align daily indicators to 6h timeframe
    roc_10_aligned = align_htf_to_ltf(prices, df_1d, roc_10)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(tc_aligned[i]) or np.isnan(bc_aligned[i]) or \
           np.isnan(roc_10_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above TC, bullish daily momentum, strong volume
            if close[i] > tc_aligned[i] and roc_10_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below BC, bearish daily momentum, strong volume
            elif close[i] < bc_aligned[i] and roc_10_aligned[i] < 0 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below BC or bearish momentum
            if close[i] < bc_aligned[i] or roc_10_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above TC or bullish momentum
            if close[i] > tc_aligned[i] or roc_10_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals