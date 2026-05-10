#!/usr/bin/env python3
# 1d_KAMA_With_1wTrend_Filter
# Hypothesis: KAMA adapts to market efficiency, providing smooth trend signals in both bull and bear markets.
# The 1-week EMA34 filter ensures trades align with the higher timeframe trend, reducing whipsaws.
# Entry occurs when price crosses KAMA with volume confirmation (1.5x average volume).
# Exit on opposite KAMA cross or trend reversal. Designed for low trade frequency (~10-25/year) to minimize fee drag.

name = "1d_KAMA_With_1wTrend_Filter"
timeframe = "1d"
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[er_len:] = change[er_len:] / volatility[er_len:]
    er[er_len:] = np.where(volatility[er_len:] == 0, 0, er[er_len:])
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-week trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation and 1w uptrend
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and
                trend_1w_up_aligned[i] > 0.5 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation and 1w downtrend
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or 1w trend turns down
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or \
               (trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or 1w trend turns up
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or \
               (trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals