#!/usr/bin/env python3
# 1d_HighLowBreakout_1wTrend_Volume
# Hypothesis: Breakouts above previous week's high (bullish) or below previous week's low (bearish) with volume confirmation and 1w trend filter (SMA50). Designed for 1d timeframe to achieve 7-25 trades/year, capturing major trend moves while avoiding whipsaws in ranging markets. Works in both bull and bear markets by following the higher timeframe trend.

name = "1d_HighLowBreakout_1wTrend_Volume"
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
    
    # 1w data for trend filter and reference levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Previous week's high and low (for breakout levels)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    # Volume confirmation: 20-period average on 1w
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1w = mean_arr(volume_1w, 20)
    
    # Align all indicators to lower timeframe (wait for 1w bar to close)
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    prev_high_aligned = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    prev_low_aligned = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for SMA50
    
    for i in range(start_idx, n):
        if np.isnan(sma_50_aligned[i]) or np.isnan(prev_high_aligned[i]) or \
           np.isnan(prev_low_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above previous week's high, above 1w SMA50, strong volume
            if close[i] > prev_high_aligned[i] and close[i] > sma_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous week's low, below 1w SMA50, strong volume
            elif close[i] < prev_low_aligned[i] and close[i] < sma_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below previous week's low or below 1w SMA50
            if close[i] < prev_low_aligned[i] or close[i] < sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above previous week's high or above 1w SMA50
            if close[i] > prev_high_aligned[i] or close[i] > sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals