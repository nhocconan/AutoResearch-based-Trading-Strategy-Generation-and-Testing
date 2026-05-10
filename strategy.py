#!/usr/bin/env python3
# 4h_VWAP_Trend_Filter_12h
# Hypothesis: 4-hour VWAP trend filter using 12-hour VWAP direction with volume confirmation.
# Uses 12-hour VWAP slope to determine trend direction, enters long when price crosses above VWAP with increasing volume,
# short when price crosses below VWAP with decreasing volume. Designed for 4h to achieve 20-50 trades/year with low frequency
# and high win rate by combining trend following with volume confirmation, suitable for both bull and bear markets.

name = "4h_VWAP_Trend_Filter_12h"
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
    
    # 12-hour data for VWAP trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate VWAP for 12h
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_numerator = np.cumsum(typical_price_12h * volume_12h)
    vwap_denominator = np.cumsum(volume_12h)
    vwap_12h = vwap_numerator / vwap_denominator
    
    # Calculate VWAP slope (5-period change) for trend direction
    vwap_slope_12h = np.full_like(vwap_12h, np.nan)
    for i in range(5, len(vwap_12h)):
        vwap_slope_12h[i] = vwap_12h[i] - vwap_12h[i-5]
    
    # Align 12h VWAP slope to 4h timeframe (wait for 12h bar to close)
    vwap_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_slope_12h)
    
    # Volume confirmation: 20-period average on 4h
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vwap_slope_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP, upward VWAP slope, increasing volume
            if close[i] > vwap_12h[-1] if len(vwap_12h) > 0 else 0 and vwap_slope_12h_aligned[i] > 0 and volume[i] > vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP, downward VWAP slope, decreasing volume
            elif close[i] < vwap_12h[-1] if len(vwap_12h) > 0 else 0 and vwap_slope_12h_aligned[i] < 0 and volume[i] < vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP or VWAP slope turns negative
            if close[i] < vwap_12h[-1] if len(vwap_12h) > 0 else 0 or vwap_slope_12h_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP or VWAP slope turns positive
            if close[i] > vwap_12h[-1] if len(vwap_12h) > 0 else 0 or vwap_slope_12h_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals